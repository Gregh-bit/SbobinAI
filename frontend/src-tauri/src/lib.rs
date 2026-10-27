use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use tauri::{Emitter, Manager, RunEvent, WindowEvent};

#[cfg(target_os = "macos")]
use window_vibrancy::{apply_vibrancy, NSVisualEffectMaterial};
#[cfg(target_os = "windows")]
use window_vibrancy::{apply_acrylic, apply_mica};

#[derive(Clone, serde::Serialize, serde::Deserialize, Debug)]
pub struct BackendStatus {
    pub status: String, // "starting", "initializing_venv", "installing_deps", "ready", "error"
    pub message: String,
    pub port_ready: bool,
    pub error: Option<String>,
}

#[derive(Clone)]
pub struct BackendState(pub Arc<Mutex<BackendStatus>>);

/// Python backend process spawned by us (None if already running or spawn failed).
pub struct BackendProcess(pub Arc<Mutex<Option<Child>>>);

#[tauri::command]
fn get_backend_status(state: tauri::State<BackendState>) -> BackendStatus {
    state.0.lock().unwrap().clone()
}

fn backend_is_up() -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], 8000).into();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn has_main(dir: &Path) -> bool {
    dir.join("main.py").is_file()
}

fn has_venv(dir: &Path) -> bool {
    let rel: &[&str] = if cfg!(windows) {
        &["venv/Scripts/python.exe", ".venv/Scripts/python.exe"]
    } else {
        &["venv/bin/python", ".venv/bin/python"]
    };
    rel.iter().any(|r| dir.join(r).is_file())
}

fn silent_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    #[allow(unused_mut)]
    let mut cmd = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

fn get_home_dir() -> Option<PathBuf> {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()
        .map(PathBuf::from)
}

fn get_full_path_env() -> String {
    let mut parts: Vec<String> = Vec::new();
    #[cfg(target_os = "macos")]
    {
        parts.push("/opt/homebrew/bin".into());
        parts.push("/opt/homebrew/sbin".into());
        parts.push("/usr/local/bin".into());
        parts.push("/usr/local/sbin".into());
    }
    if let Some(home) = get_home_dir() {
        let local_bin = home.join(".local/bin");
        if local_bin.is_dir() {
            parts.push(local_bin.to_string_lossy().to_string());
        }
        let cargo_bin = home.join(".cargo/bin");
        if cargo_bin.is_dir() {
            parts.push(cargo_bin.to_string_lossy().to_string());
        }
        #[cfg(windows)]
        {
            for ver in &["Python314", "Python313", "Python312", "Python311", "Python310", "Python39", "Python38"] {
                let py_scripts = home.join(format!(r"AppData\Local\Programs\Python\{}\Scripts", ver));
                if py_scripts.is_dir() {
                    parts.push(py_scripts.to_string_lossy().to_string());
                }
                let py_root = home.join(format!(r"AppData\Local\Programs\Python\{}", ver));
                if py_root.is_dir() {
                    parts.push(py_root.to_string_lossy().to_string());
                }
            }
        }
    }
    #[cfg(not(windows))]
    {
        parts.push("/usr/bin".into());
        parts.push("/bin".into());
        parts.push("/usr/sbin".into());
        parts.push("/sbin".into());
    }
    #[cfg(windows)]
    {
        parts.push(r"C:\Windows\System32".into());
        parts.push(r"C:\Windows".into());
        parts.push(r"C:\Windows\System32\Wbem".into());
        parts.push(r"C:\Windows\System32\WindowsPowerShell\v1.0".into());
        for ver in &["Python314", "Python313", "Python312", "Python311", "Python310", "Python39", "Python38"] {
            parts.push(format!(r"C:\Program Files\{}\Scripts", ver));
            parts.push(format!(r"C:\Program Files\{}", ver));
            parts.push(format!(r"C:\{}", ver));
        }
    }

    if let Ok(current_path) = std::env::var("PATH") {
        if !current_path.is_empty() {
            parts.push(current_path);
        }
    }
    parts.join(if cfg!(windows) { ";" } else { ":" })
}

fn python_can_run_backend(py: &Path) -> bool {
    if !py.is_file() {
        return false;
    }
    silent_command(py)
        .args(["-c", "import uvicorn, fastapi"])
        .env("PATH", get_full_path_env())
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn find_base_python() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let path_env = get_full_path_env();

    #[cfg(target_os = "macos")]
    {
        candidates.push(PathBuf::from("/opt/homebrew/bin/python3.12"));
        candidates.push(PathBuf::from("/opt/homebrew/bin/python3.11"));
        candidates.push(PathBuf::from("/opt/homebrew/bin/python3"));
        candidates.push(PathBuf::from("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"));
        candidates.push(PathBuf::from("/Library/Frameworks/Python.framework/Versions/3.11/bin/python3"));
        candidates.push(PathBuf::from("/Library/Frameworks/Python.framework/Versions/3.10/bin/python3"));
        candidates.push(PathBuf::from("/usr/local/bin/python3"));
        if let Some(home) = get_home_dir() {
            candidates.push(home.join(".pyenv/shims/python3"));
        }
        candidates.push(PathBuf::from("/usr/bin/python3"));
    }
    #[cfg(windows)]
    {
        // 1. Chiedi al Python Launcher di Windows (py.exe) il percorso dell'eseguibile Python 3 attivo
        for launcher in &["py", r"C:\Windows\py.exe"] {
            if let Ok(output) = silent_command(launcher)
                .args(["-3", "-c", "import sys; print(sys.executable)"])
                .env("PATH", &path_env)
                .output()
            {
                if output.status.success() {
                    let stdout_str = String::from_utf8_lossy(&output.stdout);
                    let trimmed = stdout_str.trim();
                    if !trimmed.is_empty() {
                        let p = PathBuf::from(trimmed);
                        if p.is_file() && !candidates.contains(&p) {
                            candidates.push(p);
                        }
                    }
                }
            }
        }

        // 2. Cerca in PATH tramite where.exe (escludendo gli alias dummy di WindowsApps che aprono lo Store)
        for tool in &["python.exe", "python3.exe", "python"] {
            if let Ok(output) = silent_command("where.exe").arg(tool).env("PATH", &path_env).output() {
                if output.status.success() {
                    for line in String::from_utf8_lossy(&output.stdout).lines() {
                        let trimmed = line.trim();
                        if !trimmed.is_empty() && !trimmed.to_lowercase().contains("windowsapps") {
                            let p = PathBuf::from(trimmed);
                            if p.is_file() && !candidates.contains(&p) {
                                candidates.push(p);
                            }
                        }
                    }
                }
            }
        }

        // 3. Cartelle standard utente in AppData (installazioni ufficiali per utente singolo)
        if let Some(home) = get_home_dir() {
            for ver in &["Python314", "Python313", "Python312", "Python311", "Python310", "Python39", "Python38"] {
                let p = home.join(format!(r"AppData\Local\Programs\Python\{}\python.exe", ver));
                if p.is_file() && !candidates.contains(&p) {
                    candidates.push(p);
                }
            }
            for conda_rel in &[
                r"miniconda3\python.exe",
                r"anaconda3\python.exe",
                r"miniforge3\python.exe",
                r".local\bin\python.exe",
                r"scoop\apps\python\current\python.exe",
            ] {
                let p = home.join(conda_rel);
                if p.is_file() && !candidates.contains(&p) {
                    candidates.push(p);
                }
            }
        }

        // 4. Cartelle standard di sistema Program Files e C:\ (installazioni per tutti gli utenti)
        for ver in &["Python314", "Python313", "Python312", "Python311", "Python310", "Python39", "Python38"] {
            for prefix in &[
                format!(r"C:\Program Files\{}\python.exe", ver),
                format!(r"C:\Program Files (x86)\{}\python.exe", ver),
                format!(r"C:\{}\python.exe", ver),
            ] {
                let p = PathBuf::from(prefix);
                if p.is_file() && !candidates.contains(&p) {
                    candidates.push(p);
                }
            }
        }
        for sys_path in &[
            r"C:\ProgramData\chocolatey\bin\python.exe",
            r"C:\ProgramData\miniconda3\python.exe",
            r"C:\ProgramData\anaconda3\python.exe",
            r"C:\miniconda3\python.exe",
            r"C:\anaconda3\python.exe",
        ] {
            let p = PathBuf::from(sys_path);
            if p.is_file() && !candidates.contains(&p) {
                candidates.push(p);
            }
        }

        // 5. WindowsApps (solo se realmente funzionante, come ultima risorsa)
        if let Some(home) = get_home_dir() {
            let p = home.join(r"AppData\Local\Microsoft\WindowsApps\python.exe");
            if p.is_file() && !candidates.contains(&p) {
                candidates.push(p);
            }
        }
    }

    for c in candidates {
        if c.is_file() {
            let ok = silent_command(&c)
                .args(["-c", "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"])
                .env("PATH", &path_env)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                log::info!("Rilevato interprete Python valido: {:?}", c);
                return Some(c);
            }
        }
    }
    None
}

fn get_app_backend_data_dir(app: &tauri::AppHandle) -> PathBuf {
    if let Ok(d) = app.path().app_local_data_dir() {
        return d.join("backend");
    }
    if let Ok(d) = app.path().app_data_dir() {
        return d.join("backend");
    }
    if let Some(home) = get_home_dir() {
        if cfg!(windows) {
            return home.join(r"AppData\Local\SbobinAI\backend");
        } else {
            return home.join("Library/Application Support/com.sbobinai.app/backend");
        }
    }
    std::env::temp_dir().join("com.sbobinai.app").join("backend")
}

fn extract_embedded_backend(target_dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(target_dir)
        .map_err(|e| format!("Impossibile creare cartella backend {:?}: {}", target_dir, e))?;
    std::fs::create_dir_all(target_dir.join("routers"))
        .map_err(|e| format!("Impossibile creare cartella routers: {}", e))?;

    macro_rules! write_embedded {
        ($rel:expr, $content:expr) => {
            std::fs::write(target_dir.join($rel), $content)
                .map_err(|e| format!("Impossibile scrivere {}: {}", $rel, e))?;
        };
    }

    write_embedded!("main.py", include_str!("../../../backend/main.py"));
    write_embedded!("config.py", include_str!("../../../backend/config.py"));
    write_embedded!("database.py", include_str!("../../../backend/database.py"));
    write_embedded!("schemas.py", include_str!("../../../backend/schemas.py"));
    write_embedded!("audio_engine.py", include_str!("../../../backend/audio_engine.py"));
    write_embedded!("transcriber.py", include_str!("../../../backend/transcriber.py"));
    write_embedded!("ai_models.py", include_str!("../../../backend/ai_models.py"));
    write_embedded!("llm_engine.py", include_str!("../../../backend/llm_engine.py"));
    write_embedded!("rag.py", include_str!("../../../backend/rag.py"));
    write_embedded!("cleaner.py", include_str!("../../../backend/cleaner.py"));
    write_embedded!("platform_utils.py", include_str!("../../../backend/platform_utils.py"));
    write_embedded!("live_session.py", include_str!("../../../backend/live_session.py"));
    write_embedded!("requirements.txt", include_str!("../../../backend/requirements.txt"));
    write_embedded!("requirements-windows.txt", include_str!("../../../backend/requirements-windows.txt"));

    write_embedded!("routers/__init__.py", include_str!("../../../backend/routers/__init__.py"));
    write_embedded!("routers/lectures.py", include_str!("../../../backend/routers/lectures.py"));
    write_embedded!("routers/models.py", include_str!("../../../backend/routers/models.py"));
    write_embedded!("routers/settings.py", include_str!("../../../backend/routers/settings.py"));
    write_embedded!("routers/study.py", include_str!("../../../backend/routers/study.py"));
    write_embedded!("routers/subjects.py", include_str!("../../../backend/routers/subjects.py"));

    let _ = std::fs::write(target_dir.join(".version"), env!("CARGO_PKG_VERSION"));
    log::info!("Estratto backend incorporato con successo (v{}) in {:?}", env!("CARGO_PKG_VERSION"), target_dir);
    Ok(())
}

fn resolve_backend_dir(app: &tauri::AppHandle) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SBOBINAI_BACKEND_DIR") {
        let p = PathBuf::from(p);
        if has_main(&p) {
            return Some(p);
        }
    }

    // 1. In dev/workspace cerca prima negli antenati una directory backend dotata di venv
    if let Ok(cwd) = std::env::current_dir() {
        for anc in cwd.ancestors().take(6) {
            let p = anc.join("backend");
            if has_main(&p) && has_venv(&p) {
                return Some(p);
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().skip(1).take(8) {
            let p = anc.join("backend");
            if has_main(&p) && has_venv(&p) {
                return Some(p);
            }
        }
    }

    // 2. Se pacchettizzata (es. App bundle macOS o NSIS Windows), controlla le cartelle risorse
    if let Ok(res) = app.path().resource_dir() {
        for candidate in &[res.join("backend"), res.join("resources/backend"), res.clone()] {
            if has_main(candidate) {
                return Some(candidate.clone());
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            for candidate in &[
                parent.join("backend"),
                parent.join("resources/backend"),
                parent.join("resources"),
            ] {
                if has_main(candidate) {
                    return Some(candidate.clone());
                }
            }
        }
    }

    // 3. Verifica cartella backend utente in AppData / Application Support:
    // Se la versione è cambiata o mancano file, ri-estrae automaticamente la versione aggiornata
    let app_data_backend = get_app_backend_data_dir(app);
    let ver_file = app_data_backend.join(".version");
    let current_ver = env!("CARGO_PKG_VERSION");
    let needs_extract = !has_main(&app_data_backend)
        || std::fs::read_to_string(&ver_file).unwrap_or_default().trim() != current_ver;
    if needs_extract {
        let _ = extract_embedded_backend(&app_data_backend);
    }
    if has_main(&app_data_backend) {
        return Some(app_data_backend);
    }

    if let Some(home) = get_home_dir() {
        let macos_p = home.join("Library/Application Support/com.sbobinai.app/backend");
        if has_main(&macos_p) {
            return Some(macos_p);
        }
        let win_p1 = home.join(r"AppData\Local\SbobinAI\backend");
        if has_main(&win_p1) {
            return Some(win_p1);
        }
        let win_p2 = home.join(r"AppData\Local\com.sbobinai.app\backend");
        if has_main(&win_p2) {
            return Some(win_p2);
        }
    }

    // 4. Fallback: qualunque directory backend che contenga main.py
    if let Ok(cwd) = std::env::current_dir() {
        for anc in cwd.ancestors().take(6) {
            let p = anc.join("backend");
            if has_main(&p) {
                return Some(p);
            }
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        for anc in exe.ancestors().skip(1).take(8) {
            let p = anc.join("backend");
            if has_main(&p) {
                return Some(p);
            }
        }
    }

    // 5. Autonomia totale: estrai i file Python incorporati direttamente nel pacchetto
    // Questo garantisce che anche gli eseguibili standalone/portatili o pacchetti privi di installer
    // funzionino al 100% senza mai dare "Directory backend mancante".
    if let Ok(()) = extract_embedded_backend(&app_data_backend) {
        if has_main(&app_data_backend) {
            return Some(app_data_backend);
        }
    }

    None
}

fn ensure_virtualenv(
    backend_dir: &Path,
    target_venv: &Path,
    state: &BackendState,
) -> Result<PathBuf, String> {
    let python_bin = if cfg!(windows) {
        target_venv.join("Scripts/python.exe")
    } else {
        target_venv.join("bin/python")
    };

    if python_can_run_backend(&python_bin) {
        return Ok(python_bin);
    }

    let req_file = if cfg!(windows) {
        backend_dir.join("requirements-windows.txt")
    } else {
        backend_dir.join("requirements.txt")
    };
    if !req_file.is_file() {
        return Err(format!("File dei requisiti non trovato in {:?}", req_file));
    }

    let path_env = get_full_path_env();

    // 1. PRIORITÀ 1: Se l'utente ha già un interprete Python sul proprio computer, usa SEMPRE quello!
    // Evita di riscaricare Python o aprire finestre inutili.
    if let Some(base_py) = find_base_python() {
        log::info!("Trovato interprete Python utente già presente sul sistema: {:?}", base_py);
        {
            let mut s = state.0.lock().unwrap();
            s.status = "initializing_venv".into();
            s.message = "Configurazione dell'ambiente locale in corso...".into();
        }

        let _ = std::fs::create_dir_all(target_venv.parent().unwrap_or(target_venv));

        // Crea il venv tramite python -m venv in modalità completamente silenziosa (senza finestre cmd)
        let venv_status = silent_command(&base_py)
            .args(["-m", "venv", target_venv.to_str().unwrap_or_default()])
            .env("PATH", &path_env)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Errore creazione venv con {:?}: {e}", base_py))?;

        if venv_status.success() {
            {
                let mut s = state.0.lock().unwrap();
                s.status = "installing_deps".into();
                s.message = "Installazione delle librerie del motore in corso (richiede qualche minuto al primo avvio)...".into();
            }

            let pip_bin = if cfg!(windows) {
                target_venv.join("Scripts/pip.exe")
            } else {
                target_venv.join("bin/pip")
            };

            let mut pip_args: Vec<&str> = vec!["install", "--no-warn-script-location", "-r", req_file.to_str().unwrap_or_default()];
            if cfg!(windows) {
                pip_args.push("--extra-index-url");
                pip_args.push("https://abetlen.github.io/llama-cpp-python/whl/cpu");
            }

            let pip_status = silent_command(&pip_bin)
                .args(&pip_args)
                .env("PATH", &path_env)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map_err(|e| format!("Errore installazione dipendenze: {e}"))?;

            if pip_status.success() && python_can_run_backend(&python_bin) {
                log::info!("Ambiente configurato con successo usando Python di sistema: {:?}", python_bin);
                return Ok(python_bin);
            }
        }
        log::warn!("Configurazione con Python di sistema non completata, passo al fallback autonomo uv");
    }

    // 2. PRIORITÀ 2 (Fallback solo se l'utente non ha alcun Python sul PC):
    // Scarica silenziosamente uv e configura il runtime autonomo 3.12 senza alcuna finestra cmd a vista
    let mut uv_bin = if cfg!(windows) {
        PathBuf::from("uv.exe")
    } else {
        PathBuf::from("uv")
    };
    if let Some(home) = get_home_dir() {
        #[cfg(windows)]
        {
            for candidate in &[
                home.join(r".local\bin\uv.exe"),
                home.join(r"AppData\Local\Programs\uv\uv.exe"),
                home.join(r"AppData\Local\uv\uv.exe"),
                home.join(r"AppData\Roaming\uv\uv.exe"),
                home.join(r".cargo\bin\uv.exe"),
            ] {
                if candidate.is_file() {
                    uv_bin = candidate.clone();
                    break;
                }
            }
        }
        #[cfg(not(windows))]
        {
            let local_uv = home.join(".local/bin/uv");
            if local_uv.is_file() {
                uv_bin = local_uv;
            } else {
                let cargo_uv = home.join(".cargo/bin/uv");
                if cargo_uv.is_file() {
                    uv_bin = cargo_uv;
                }
            }
        }
    }

    let mut has_uv = silent_command(&uv_bin)
        .arg("--version")
        .env("PATH", &path_env)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !has_uv {
        #[cfg(not(windows))]
        {
            {
                let mut s = state.0.lock().unwrap();
                s.status = "initializing_venv".into();
                s.message = "Sto preparando l'ambiente per l'uso, aspetta ancora un po'...".into();
            }
            let _ = silent_command("/bin/sh")
                .args(["-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"])
                .env("PATH", &path_env)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            if let Some(home) = get_home_dir() {
                let local_uv = home.join(".local/bin/uv");
                if local_uv.is_file() {
                    uv_bin = local_uv;
                    has_uv = true;
                }
            }
        }
        #[cfg(windows)]
        {
            {
                let mut s = state.0.lock().unwrap();
                s.status = "initializing_venv".into();
                s.message = "Configurazione componenti in corso...".into();
            }
            let _ = silent_command("powershell")
                .args(["-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "irm https://astral.sh/uv/install.ps1 | iex"])
                .env("PATH", &path_env)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();

            if let Some(home) = get_home_dir() {
                for candidate in &[
                    home.join(r".local\bin\uv.exe"),
                    home.join(r"AppData\Local\Programs\uv\uv.exe"),
                    home.join(r"AppData\Local\uv\uv.exe"),
                    home.join(r"AppData\Roaming\uv\uv.exe"),
                    home.join(r".cargo\bin\uv.exe"),
                ] {
                    if candidate.is_file() {
                        uv_bin = candidate.clone();
                        has_uv = true;
                        break;
                    }
                }
            }
        }
    }

    if has_uv {
        {
            let mut s = state.0.lock().unwrap();
            s.status = "initializing_venv".into();
            s.message = "Configurazione runtime locale autonomo in corso...".into();
        }
        let _ = std::fs::create_dir_all(target_venv.parent().unwrap_or(target_venv));
        let status = silent_command(&uv_bin)
            .args(["venv", target_venv.to_str().unwrap_or_default(), "--python", "3.12"])
            .env("PATH", &path_env)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Errore configurazione ambiente uv: {e}"))?;
        if !status.success() {
            return Err("Configurazione ambiente uv fallita".into());
        }

        {
            let mut s = state.0.lock().unwrap();
            s.status = "installing_deps".into();
            s.message = "Installazione delle librerie del motore in corso (richiede qualche minuto al primo avvio)...".into();
        }
        let mut pip_args: Vec<&str> = vec![
            "pip",
            "install",
            "-r",
            req_file.to_str().unwrap_or_default(),
            "--python",
            python_bin.to_str().unwrap_or_default(),
        ];
        if cfg!(windows) {
            pip_args.push("--extra-index-url");
            pip_args.push("https://abetlen.github.io/llama-cpp-python/whl/cpu");
        }
        let status = silent_command(&uv_bin)
            .args(&pip_args)
            .env("PATH", &path_env)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|e| format!("Errore configurazione componenti uv: {e}"))?;
        if !status.success() {
            return Err("Configurazione componenti uv fallita".into());
        }
    } else {
        return Err("Impossibile avviare il motore. Assicurati che Python sia installato.".into());
    }

    if python_can_run_backend(&python_bin) {
        Ok(python_bin)
    } else {
        Err("L'ambiente Python creato non ha convalidato le dipendenze uvicorn/fastapi.".into())
    }
}

fn run_backend_bootstrap(
    app: tauri::AppHandle,
    state: BackendState,
    process: BackendProcess,
) {
    if backend_is_up() {
        log::info!("Backend already listening on 127.0.0.1:8000");
        let mut s = state.0.lock().unwrap();
        s.status = "ready".into();
        s.message = "Motore locale SbobinAI pronto".into();
        s.port_ready = true;
        return;
    }

    let backend_dir = match resolve_backend_dir(&app) {
        Some(d) => d,
        None => {
            log::error!("Cartella backend non trovata");
            let mut s = state.0.lock().unwrap();
            s.status = "error".into();
            s.message = "Cartella backend non trovata nel pacchetto applicativo.".into();
            s.error = Some("Directory backend mancante".into());
            return;
        }
    };

    // 1. Cerca prima se esiste già un venv di sviluppo con dipendenze attive
    let dev_py = if cfg!(windows) {
        backend_dir.join("venv/Scripts/python.exe")
    } else {
        backend_dir.join("venv/bin/python")
    };

    let python = if python_can_run_backend(&dev_py) {
        log::info!("Uso venv locale di sviluppo verificato: {:?}", dev_py);
        dev_py
    } else {
        // 2. Altrimenti usa o configura il venv utente persistente
        let home = match get_home_dir() {
            Some(h) => h,
            None => {
                let mut s = state.0.lock().unwrap();
                s.status = "error".into();
                s.message = "Impossibile determinare la cartella home dell'utente.".into();
                return;
            }
        };

        let target_venv = if cfg!(target_os = "macos") {
            home.join("Library/Application Support/com.sbobinai.app/venv")
        } else if cfg!(windows) {
            home.join(r"AppData\Local\SbobinAI\venv")
        } else {
            home.join(".sbobinai/venv")
        };

        match ensure_virtualenv(&backend_dir, &target_venv, &state) {
            Ok(py) => py,
            Err(e) => {
                log::error!("Errore bootstrap venv: {}", e);
                let mut s = state.0.lock().unwrap();
                s.status = "error".into();
                s.message = "Errore durante la preparazione dell'ambiente AI locale.".into();
                s.error = Some(e);
                return;
            }
        }
    };

    {
        let mut s = state.0.lock().unwrap();
        s.status = "starting".into();
        s.message = "Avvio del motore AI locale in corso...".into();
    }

    let path_env = get_full_path_env();
    let mut cmd = silent_command(&python);
    cmd.args(["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&backend_dir)
        .env("PATH", &path_env)
        .env("PYTHONPATH", &backend_dir)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::null());

    #[cfg(not(debug_assertions))]
    let log_file = get_home_dir().and_then(|home| {
        #[cfg(target_os = "macos")]
        let log_dir = home.join("Library/Logs/SbobinAI");
        #[cfg(not(target_os = "macos"))]
        let log_dir = home.join(".sbobinai/logs");
        let _ = std::fs::create_dir_all(&log_dir);
        let path = log_dir.join("backend.log");
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()
    });

    #[cfg(debug_assertions)]
    {
        cmd.stdout(Stdio::inherit()).stderr(Stdio::inherit());
    }
    #[cfg(not(debug_assertions))]
    {
        if let Some(ref file) = log_file {
            if let Ok(f_err) = file.try_clone() {
                cmd.stdout(file.try_clone().unwrap_or_else(|_| f_err));
                if let Ok(f2) = file.try_clone() {
                    cmd.stderr(f2);
                } else {
                    cmd.stderr(Stdio::null());
                }
            } else {
                cmd.stdout(Stdio::null()).stderr(Stdio::null());
            }
        } else {
            cmd.stdout(Stdio::null()).stderr(Stdio::null());
        }
    }

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }

    match cmd.spawn() {
        Ok(child) => {
            log::info!("Backend avviato con successo (pid {})", child.id());
            if let Ok(mut g) = process.0.lock() {
                *g = Some(child);
            }

            for i in 0..60 {
                if backend_is_up() {
                    log::info!("Porta 8000 attiva e verificata dopo {} cicli", i + 1);
                    let mut s = state.0.lock().unwrap();
                    s.status = "ready".into();
                    s.message = "Motore locale SbobinAI pronto".into();
                    s.port_ready = true;
                    return;
                }
                std::thread::sleep(Duration::from_millis(500));
            }

            let log_hint = if cfg!(windows) {
                r"%LOCALAPPDATA%\SbobinAI\logs\backend.log"
            } else {
                "~/Library/Logs/SbobinAI/backend.log"
            };
            let mut s = state.0.lock().unwrap();
            s.status = "error".into();
            s.message = format!("Il motore locale non ha risposto entro 30s. Controlla {log_hint}.");
            s.error = Some("Timeout avvio server locale".into());
        }
        Err(e) => {
            log::error!("Errore avvio comando uvicorn: {}", e);
            let mut s = state.0.lock().unwrap();
            s.status = "error".into();
            s.message = format!("Impossibile avviare il backend: {e}");
            s.error = Some(e.to_string());
        }
    }
}

fn kill_backend(app: &tauri::AppHandle) {
    let Some(state) = app.try_state::<BackendProcess>() else {
        return;
    };
    let child = match state.0.lock() {
        Ok(mut g) => g.take(),
        Err(_) => None,
    };
    if let Some(mut child) = child {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &child.id().to_string()])
                .creation_flags(0x0800_0000)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &child.id().to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            std::thread::sleep(Duration::from_millis(300));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[cfg(target_os = "macos")]
fn apply_macos_dock_icon() {
    use objc2::AnyThread;
    use objc2::MainThreadMarker;
    use objc2_app_kit::{NSApplication, NSImage};
    use objc2_foundation::NSData;

    static ICON_BYTES: &[u8] = include_bytes!("../icons/icon.png");

    if let Some(mtm) = MainThreadMarker::new() {
        let ns_data = NSData::with_bytes(ICON_BYTES);
        let ns_image = NSImage::initWithData(NSImage::alloc(), &ns_data);
        if let Some(image) = ns_image {
            let app = NSApplication::sharedApplication(mtm);
            unsafe {
                app.setApplicationIconImage(Some(&image));
            }
            log::info!("Applied macOS dynamic dock icon");
        }
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let backend_status = Arc::new(Mutex::new(BackendStatus {
        status: "starting".into(),
        message: "Avvio del motore locale SbobinAI...".into(),
        port_ready: false,
        error: None,
    }));
    let backend_proc_arc = Arc::new(Mutex::new(None));

    let backend_state = BackendState(backend_status);
    let backend_proc = BackendProcess(backend_proc_arc.clone());

    let state_for_setup = backend_state.clone();
    let proc_for_setup = BackendProcess(backend_proc_arc);

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(backend_state)
        .manage(backend_proc)
        .invoke_handler(tauri::generate_handler![get_backend_status])
        .setup(move |app| {
            #[cfg(target_os = "macos")]
            apply_macos_dock_icon();

            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let app_handle = app.handle().clone();
            std::thread::spawn(move || {
                run_backend_bootstrap(app_handle, state_for_setup, proc_for_setup);
            });

            // Window effects
            #[cfg(any(target_os = "macos", target_os = "windows"))]
            if let Some(window) = app.get_webview_window("main") {
                #[cfg(target_os = "macos")]
                if let Err(e) =
                    apply_vibrancy(&window, NSVisualEffectMaterial::HudWindow, None, None)
                {
                    log::warn!("apply_vibrancy failed: {e}");
                }
                #[cfg(target_os = "windows")]
                if let Err(e) = apply_mica(&window, None) {
                    log::warn!("apply_mica failed ({e}), trying acrylic");
                    if let Err(e) = apply_acrylic(&window, Some((18, 18, 18, 125))) {
                        log::warn!("apply_acrylic failed: {e}");
                    }
                }
            }

            // Desktop-only plugins
            #[cfg(desktop)]
            {
                use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

                app.handle()
                    .plugin(tauri_plugin_updater::Builder::new().build())?;

                app.handle().plugin(
                    tauri_plugin_global_shortcut::Builder::new()
                        .with_handler(|app, _shortcut, event| {
                            if event.state() == ShortcutState::Pressed {
                                if let Err(e) = app.emit("global-bookmark", ()) {
                                    log::warn!("emit global-bookmark failed: {e}");
                                }
                            }
                        })
                        .build(),
                )?;
                if let Err(e) = app.global_shortcut().register("CommandOrControl+Shift+B") {
                    log::warn!("global shortcut registration failed: {e}");
                }
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app, event| match event {
        RunEvent::WindowEvent {
            label,
            event: WindowEvent::Destroyed,
            ..
        } if label == "main" => kill_backend(app),
        RunEvent::Exit => kill_backend(app),
        _ => {}
    });
}
