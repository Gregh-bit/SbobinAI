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
    }
    parts.push("/usr/bin".into());
    parts.push("/bin".into());
    parts.push("/usr/sbin".into());
    parts.push("/sbin".into());

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
    Command::new(py)
        .args(["-c", "import uvicorn, fastapi"])
        .env("PATH", get_full_path_env())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn find_base_python() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
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
        if let Some(home) = get_home_dir() {
            candidates.push(home.join(r"AppData\Local\Programs\Python\Python312\python.exe"));
            candidates.push(home.join(r"AppData\Local\Programs\Python\Python311\python.exe"));
            candidates.push(home.join(r"AppData\Local\Programs\Python\Python310\python.exe"));
        }
        candidates.push(PathBuf::from(r"C:\Program Files\Python312\python.exe"));
        candidates.push(PathBuf::from(r"C:\Program Files\Python311\python.exe"));
        candidates.push(PathBuf::from(r"C:\Python312\python.exe"));
        candidates.push(PathBuf::from("python.exe"));
    }
    for c in candidates {
        if c.is_file() {
            let ok = Command::new(&c)
                .args(["-c", "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"])
                .env("PATH", get_full_path_env())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                return Some(c);
            }
        }
    }
    None
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

    // 2. Se pacchettizzata (es. App bundle), controlla la cartella risorse
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("backend");
        if has_main(&p) {
            return Some(p);
        }
    }

    // 3. Verifica percorsi workspace comuni nel profilo utente
    if let Some(home) = get_home_dir() {
        for rel in &[
            "Desktop/prova1/backend",
            "Documents/SbobinAI/backend",
            "SbobinAI/backend",
            "Library/Application Support/com.sbobinai.app/backend",
        ] {
            let p = home.join(rel);
            if has_main(&p) {
                return Some(p);
            }
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
    let mut uv_bin = PathBuf::from("uv");
    if let Some(home) = get_home_dir() {
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

    let mut has_uv = Command::new(&uv_bin)
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
                s.message = "Configurazione motore AI portatile (uv)...".into();
            }
            let _ = Command::new("/bin/sh")
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
    }

    if has_uv {
        {
            let mut s = state.0.lock().unwrap();
            s.status = "initializing_venv".into();
            s.message = "Configurazione ambiente Python 3.12 con uv...".into();
        }
        let _ = std::fs::create_dir_all(target_venv.parent().unwrap_or(target_venv));
        let status = Command::new(&uv_bin)
            .args(["venv", target_venv.to_str().unwrap_or_default(), "--python", "3.12"])
            .env("PATH", &path_env)
            .status()
            .map_err(|e| format!("Errore esecuzione uv venv: {e}"))?;
        if !status.success() {
            return Err("Fallito comando uv venv".into());
        }

        {
            let mut s = state.0.lock().unwrap();
            s.status = "installing_deps".into();
            s.message = "Installazione pacchetti AI locali con uv...".into();
        }
        let status = Command::new(&uv_bin)
            .args([
                "pip",
                "install",
                "-r",
                req_file.to_str().unwrap_or_default(),
                "--python",
                python_bin.to_str().unwrap_or_default(),
            ])
            .env("PATH", &path_env)
            .status()
            .map_err(|e| format!("Errore esecuzione uv pip install: {e}"))?;
        if !status.success() {
            return Err("Fallito comando uv pip install".into());
        }
    } else {
        let base_py = find_base_python().ok_or_else(|| {
            "Nessun interprete Python 3 trovato sul sistema. Assicurati che Python sia installato.".to_string()
        })?;

        {
            let mut s = state.0.lock().unwrap();
            s.status = "initializing_venv".into();
            s.message = "Inizializzazione ambiente Python in corso...".into();
        }
        let _ = std::fs::create_dir_all(target_venv.parent().unwrap_or(target_venv));
        let status = Command::new(&base_py)
            .args(["-m", "venv", target_venv.to_str().unwrap_or_default()])
            .env("PATH", &path_env)
            .status()
            .map_err(|e| format!("Errore creazione venv con {:?}: {e}", base_py))?;
        if !status.success() {
            return Err("Fallita creazione virtualenv con Python di sistema".into());
        }

        {
            let mut s = state.0.lock().unwrap();
            s.status = "installing_deps".into();
            s.message = "Installazione dipendenze AI al primo avvio (richiede ~1 min)...".into();
        }
        let pip_bin = if cfg!(windows) {
            target_venv.join("Scripts/pip.exe")
        } else {
            target_venv.join("bin/pip")
        };
        let status = Command::new(&pip_bin)
            .args(["install", "-r", req_file.to_str().unwrap_or_default()])
            .env("PATH", &path_env)
            .status()
            .map_err(|e| format!("Errore installazione dipendenze con pip: {e}"))?;
        if !status.success() {
            return Err("Fallita installazione dipendenze tramite pip".into());
        }
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
    let mut cmd = Command::new(&python);
    cmd.args(["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&backend_dir)
        .env("PATH", &path_env)
        .env("PYTHONPATH", &backend_dir)
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

            let mut s = state.0.lock().unwrap();
            s.status = "error".into();
            s.message = "Il motore locale non ha risposto entro 30s. Controlla ~/Library/Logs/SbobinAI/backend.log.".into();
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
        let _ = child.kill();
        let _ = child.wait();
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
