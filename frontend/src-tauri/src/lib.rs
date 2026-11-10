use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Token di cancellazione atomico per arrestare immediatamente il bootstrap del backend alla chiusura dell'app.
#[derive(Clone)]
pub struct AppShutdown(pub Arc<AtomicBool>);

// ── Native OS Wake Lock (Anti-sleep / Sleep Prevention) ─────────────────────────

#[cfg(target_os = "macos")]
mod macos_power {
    use std::ffi::c_void;

    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOPMAssertionCreateWithName(
            assertion_type: *const c_void,
            assertion_level: u32,
            assertion_name: *const c_void,
            assertion_id: *mut u32,
        ) -> i32;

        fn IOPMAssertionRelease(assertion_id: u32) -> i32;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(
            alloc: *const c_void,
            c_str: *const std::ffi::c_char,
            encoding: u32,
        ) -> *const c_void;
        fn CFRelease(cf: *const c_void);
    }

    const K_CF_STRING_ENCODING_UTF8: u32 = 0x08000100;
    const K_IOPM_ASSERTION_LEVEL_ON: u32 = 255;

    pub fn create_wake_lock(reason: &str) -> Option<u32> {
        unsafe {
            let type_str = std::ffi::CString::new("PreventUserIdleSystemSleep").ok()?;
            let reason_str = std::ffi::CString::new(reason).ok()?;

            let cf_type = CFStringCreateWithCString(std::ptr::null(), type_str.as_ptr(), K_CF_STRING_ENCODING_UTF8);
            let cf_reason = CFStringCreateWithCString(std::ptr::null(), reason_str.as_ptr(), K_CF_STRING_ENCODING_UTF8);

            if cf_type.is_null() || cf_reason.is_null() {
                if !cf_type.is_null() {
                    CFRelease(cf_type);
                }
                if !cf_reason.is_null() {
                    CFRelease(cf_reason);
                }
                return None;
            }

            let mut assertion_id: u32 = 0;
            let ret = IOPMAssertionCreateWithName(
                cf_type,
                K_IOPM_ASSERTION_LEVEL_ON,
                cf_reason,
                &mut assertion_id,
            );

            CFRelease(cf_type);
            CFRelease(cf_reason);

            if ret == 0 {
                log::info!("[WakeLock] Asserzione macOS IOKit creata con successo (ID: {})", assertion_id);
                Some(assertion_id)
            } else {
                log::warn!("[WakeLock] Errore creazione asserzione IOKit: {}", ret);
                None
            }
        }
    }

    pub fn release_wake_lock(assertion_id: u32) {
        unsafe {
            let ret = IOPMAssertionRelease(assertion_id);
            if ret == 0 {
                log::info!("[WakeLock] Asserzione macOS IOKit rilasciata (ID: {})", assertion_id);
            } else {
                log::warn!("[WakeLock] Errore rilascio asserzione IOKit {}: {}", assertion_id, ret);
            }
        }
    }
}

#[cfg(target_os = "windows")]
mod windows_power {
    type ExecutionState = u32;
    const ES_CONTINUOUS: ExecutionState = 0x80000000;
    const ES_SYSTEM_REQUIRED: ExecutionState = 0x00000001;
    const ES_AWAYMODE_REQUIRED: ExecutionState = 0x00000040;

    unsafe extern "system" {
        fn SetThreadExecutionState(es_flags: ExecutionState) -> ExecutionState;
    }

    pub fn create_wake_lock() -> bool {
        unsafe {
            // Tentativo 1: Continuous + System Required + Away Mode (supportato da Windows Vista/7/8/10/11)
            let prev = SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED);
            if prev == 0 {
                // Tentativo 2: Fallback se Away Mode non è supportato dall'hardware/versione di Windows
                let prev_fallback = SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
                if prev_fallback == 0 {
                    log::warn!("[WakeLock] SetThreadExecutionState fallito su Windows");
                    return false;
                }
            }
            log::info!("[WakeLock] Anti-sleep attivato con successo su Windows (SetThreadExecutionState)");
            true
        }
    }

    pub fn release_wake_lock() {
        unsafe {
            SetThreadExecutionState(ES_CONTINUOUS);
            log::info!("[WakeLock] Anti-sleep disattivato su Windows (ripristinato ES_CONTINUOUS)");
        }
    }
}

#[derive(Default)]
pub struct WakeLockState {
    #[cfg(target_os = "macos")]
    pub assertion_id: Mutex<Option<u32>>,
    #[cfg(target_os = "windows")]
    pub active: AtomicBool,
}

#[tauri::command]
fn acquire_wake_lock(state: tauri::State<WakeLockState>, reason: Option<String>) -> Result<(), String> {
    let r = reason.unwrap_or_else(|| "SbobinAI - Registrazione audio lezione in corso".to_string());
    #[cfg(target_os = "macos")]
    {
        let mut guard = state.assertion_id.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_none() {
            if let Some(id) = macos_power::create_wake_lock(&r) {
                *guard = Some(id);
            }
        }
    }
    #[cfg(target_os = "windows")]
    {
        if !state.active.swap(true, Ordering::SeqCst) {
            windows_power::create_wake_lock();
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = r;
    }
    Ok(())
}

#[tauri::command]
fn release_wake_lock(state: tauri::State<WakeLockState>) -> Result<(), String> {
    #[cfg(target_os = "macos")]
    {
        let mut guard = state.assertion_id.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(id) = guard.take() {
            macos_power::release_wake_lock(id);
        }
    }
    #[cfg(target_os = "windows")]
    {
        if state.active.swap(false, Ordering::SeqCst) {
            windows_power::release_wake_lock();
        }
    }
    Ok(())
}

#[tauri::command]
fn get_backend_status(state: tauri::State<BackendState>) -> BackendStatus {
    state.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
}

#[tauri::command]
fn stop_backend_for_update(app: tauri::AppHandle) {
    request_http_shutdown();
    kill_backend(&app);
    terminate_stale_backend_on_port_8000();
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum BackendHealth {
    Down,
    MatchingVersion,
    StaleVersion,
}

fn probe_backend_health() -> BackendHealth {
    let addr: SocketAddr = ([127, 0, 0, 1], 8000).into();
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(400)) else {
        return BackendHealth::Down;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(600)));
    let _ = stream.set_write_timeout(Some(Duration::from_millis(400)));

    let req = b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nConnection: close\r\n\r\n";
    if stream.write_all(req).is_err() {
        return BackendHealth::Down;
    }

    let mut buf = [0u8; 1024];
    match stream.read(&mut buf) {
        Ok(n) if n > 0 => {
            let resp = String::from_utf8_lossy(&buf[..n]);
            if !resp.starts_with("HTTP/1.1 200")
                || !(resp.contains("SbobinAI") || (resp.contains("status") && resp.contains("ok")))
            {
                return BackendHealth::Down;
            }
            if cfg!(debug_assertions) {
                return BackendHealth::MatchingVersion;
            }
            if let Some((_, body)) = resp.split_once("\r\n\r\n") {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(body.trim()) {
                    if json.get("app_version").and_then(|v| v.as_str()) == Some(env!("CARGO_PKG_VERSION")) {
                        return BackendHealth::MatchingVersion;
                    }
                }
            }
            BackendHealth::StaleVersion
        }
        _ => BackendHealth::Down,
    }
}

fn backend_is_up() -> bool {
    probe_backend_health() != BackendHealth::Down
}

fn request_http_shutdown() {
    let addr: SocketAddr = ([127, 0, 0, 1], 8000).into();
    if let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(300)) {
        let _ = stream.set_write_timeout(Some(Duration::from_millis(300)));
        let _ = stream.set_read_timeout(Some(Duration::from_millis(400)));
        let req = b"POST /api/v1/system/shutdown HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(req);
        let mut buf = [0u8; 128];
        let _ = stream.read(&mut buf);
    }
}

fn terminate_stale_backend_on_port_8000() {
    log::warn!("Trovato backend SbobinAI di una versione precedente su 127.0.0.1:8000: arresto forzato...");
    request_http_shutdown();
    std::thread::sleep(Duration::from_millis(350));

    if backend_is_up() {
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            let _ = Command::new("powershell")
                .args([
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { $p = Get-Process -Id $_ -ErrorAction SilentlyContinue; if ($p -and ($p.ProcessName -like 'python*' -or $p.ProcessName -like 'uvicorn*')) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } }",
                ])
                .creation_flags(CREATE_NO_WINDOW)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        #[cfg(unix)]
        {
            let _ = Command::new("sh")
                .args([
                    "-c",
                    "for p in $(lsof -tiTCP:8000 -sTCP:LISTEN 2>/dev/null); do comm=$(ps -p $p -o comm= 2>/dev/null); case \"$comm\" in *python*|*Python*|*uvicorn*) kill -TERM $p 2>/dev/null;; esac; done",
                ])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }

    for _ in 0..10 {
        if !backend_is_up() {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
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
        for prefix in &["/opt/homebrew/bin", "/usr/local/bin"] {
            for ver in &["python3.12", "python3.11", "python3.10", "python3"] {
                let p = PathBuf::from(format!("{prefix}/{ver}"));
                if p.is_file() && !candidates.contains(&p) {
                    candidates.push(p);
                }
            }
        }
        for ver in &["3.12", "3.11", "3.10"] {
            let p = PathBuf::from(format!("/Library/Frameworks/Python.framework/Versions/{ver}/bin/python3"));
            if p.is_file() && !candidates.contains(&p) {
                candidates.push(p);
            }
        }
        if let Some(home) = get_home_dir() {
            candidates.push(home.join(".pyenv/shims/python3"));
            candidates.push(home.join(".local/bin/python3"));
        }
        // Scansione dinamica del PATH utente
        if let Ok(output) = Command::new("/usr/bin/which").arg("python3").output() {
            if output.status.success() {
                let trimmed = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if !trimmed.is_empty() {
                    let p = PathBuf::from(trimmed);
                    if p.is_file() && !candidates.contains(&p) && !p.starts_with("/usr/bin") {
                        candidates.push(p);
                    }
                }
            }
        }
        // /usr/bin/python3 solo se Command Line Tools di Xcode sono installati (evita popup di sistema)
        let has_clt = Command::new("/usr/bin/xcode-select")
            .arg("-p")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if has_clt {
            let p = PathBuf::from("/usr/bin/python3");
            if p.is_file() && !candidates.contains(&p) {
                candidates.push(p);
            }
        }
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
                .args(["-c", "import sys; exit(0 if (3, 10) <= sys.version_info < (3, 13) else 1)"])
                .env("PATH", &path_env)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if ok {
                log::info!("Rilevato interprete Python valido (3.10-3.12): {:?}", c);
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
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
                let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
                let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
                for cand in &[home.join(".local/bin/uv"), home.join(".cargo/bin/uv")] {
                    if cand.is_file() {
                        uv_bin = cand.clone();
                        has_uv = true;
                        break;
                    }
                }
            }
        }
        #[cfg(windows)]
        {
            {
                let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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
    shutdown: Arc<AtomicBool>,
) {
    if shutdown.load(Ordering::SeqCst) {
        return;
    }

    match probe_backend_health() {
        BackendHealth::MatchingVersion => {
            log::info!("Backend v{} already listening on 127.0.0.1:8000", env!("CARGO_PKG_VERSION"));
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
            s.status = "ready".into();
            s.message = "Motore locale SbobinAI pronto".into();
            s.port_ready = true;
            return;
        }
        BackendHealth::StaleVersion => {
            terminate_stale_backend_on_port_8000();
        }
        BackendHealth::Down => {}
    }

    let backend_dir = match resolve_backend_dir(&app) {
        Some(d) => d,
        None => {
            log::error!("Cartella backend non trovata");
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
            s.status = "error".into();
            s.message = "Cartella backend non trovata nel pacchetto applicativo.".into();
            s.error = Some("Directory backend mancante".into());
            return;
        }
    };

    if shutdown.load(Ordering::SeqCst) {
        return;
    }

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
                let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
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

        if shutdown.load(Ordering::SeqCst) {
            return;
        }

        match ensure_virtualenv(&backend_dir, &target_venv, &state) {
            Ok(py) => py,
            Err(e) => {
                log::error!("Errore bootstrap venv: {}", e);
                let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
                s.status = "error".into();
                s.message = "Errore durante la preparazione dell'ambiente AI locale.".into();
                s.error = Some(e);
                return;
            }
        }
    };

    if shutdown.load(Ordering::SeqCst) {
        log::info!("Bootstrap backend interrotto per shutdown applicazione.");
        return;
    }

    {
        let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
        s.status = "starting".into();
        s.message = "Avvio del motore AI locale in corso...".into();
    }

    match probe_backend_health() {
        BackendHealth::MatchingVersion => {
            log::info!("Backend v{} già attivo prima dello spawn", env!("CARGO_PKG_VERSION"));
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
            s.status = "ready".into();
            s.message = "Motore locale SbobinAI pronto".into();
            s.port_ready = true;
            return;
        }
        BackendHealth::StaleVersion => {
            terminate_stale_backend_on_port_8000();
        }
        BackendHealth::Down => {}
    }

    let path_env = get_full_path_env();
    let mut cmd = silent_command(&python);
    cmd.args(["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
        .current_dir(&backend_dir)
        .env("PATH", &path_env)
        .env("PYTHONPATH", &backend_dir)
        .env("PYTHONUTF8", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("SBOBINAI_APP_VERSION", env!("CARGO_PKG_VERSION"))
        .stdin(Stdio::null());

    #[cfg(not(debug_assertions))]
    let log_file = get_home_dir().and_then(|home| {
        #[cfg(target_os = "macos")]
        let log_dir = home.join("Library/Logs/SbobinAI");
        #[cfg(target_os = "windows")]
        let log_dir = std::env::var("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home.join(r"AppData\Local"))
            .join("SbobinAI")
            .join("logs");
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
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
            if shutdown.load(Ordering::SeqCst) {
                let mut child = child;
                let _ = child.kill();
                return;
            }
            log::info!("Backend avviato con successo (pid {})", child.id());
            if let Ok(mut g) = process.0.lock() {
                *g = Some(child);
            }

            for i in 0..60 {
                if shutdown.load(Ordering::SeqCst) {
                    return;
                }
                if backend_is_up() {
                    log::info!("Porta 8000 attiva e verificata dopo {} cicli", i + 1);
                    let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
                    s.status = "ready".into();
                    s.message = "Motore locale SbobinAI pronto".into();
                    s.port_ready = true;
                    return;
                }

                // Controlla se il child process è terminato prematuramente (es. crash all'import o porta occupata)
                if let Ok(mut g) = process.0.lock() {
                    if let Some(ref mut child_proc) = *g {
                        if let Ok(Some(exit_status)) = child_proc.try_wait() {
                            log::error!("Il backend Python è terminato inaspettatamente con stato: {exit_status}");
                            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
                            s.status = "error".into();
                            s.message = format!("Il motore locale si è arrestato all'avvio ({exit_status}).");
                            s.error = Some(format!("Crash processo backend: {exit_status}"));
                            return;
                        }
                    }
                }

                std::thread::sleep(Duration::from_millis(500));
            }

            let log_hint = if cfg!(windows) {
                r"%LOCALAPPDATA%\SbobinAI\logs\backend.log"
            } else {
                "~/Library/Logs/SbobinAI/backend.log"
            };
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
            s.status = "error".into();
            s.message = format!("Il motore locale non ha risposto entro 30s. Controlla {log_hint}.");
            s.error = Some("Timeout avvio server locale".into());
        }
        Err(e) => {
            log::error!("Errore avvio comando uvicorn: {}", e);
            let mut s = state.0.lock().unwrap_or_else(|p| p.into_inner());
            s.status = "error".into();
            s.message = format!("Impossibile avviare il backend: {e}");
            s.error = Some(e.to_string());
        }
    }
}

fn kill_backend(app: &tauri::AppHandle) {
    if let Some(shutdown) = app.try_state::<AppShutdown>() {
        shutdown.0.store(true, Ordering::SeqCst);
    }
    let Some(state) = app.try_state::<BackendProcess>() else {
        return;
    };
    let child = match state.0.lock() {
        Ok(mut g) => g.take(),
        Err(poisoned) => poisoned.into_inner().take(),
    };
    if let Some(mut child) = child {
        let pid = child.id();
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            let _ = Command::new("taskkill")
                .args(["/T", "/F", "/PID", &pid.to_string()])
                .creation_flags(0x0800_0000)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &format!("-{}", pid)])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            let _ = Command::new("kill")
                .args(["-TERM", &pid.to_string()])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            std::thread::sleep(Duration::from_millis(800));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

#[tauri::command]
async fn uninstall_sbobinai(
    app: tauri::AppHandle,
    keep_storage_path: Option<String>,
    delete_storage_folder: Option<bool>,
) -> Result<(), String> {
    let should_delete_storage = delete_storage_folder.unwrap_or(false);
    log::info!("Richiesta disinstallazione SbobinAI avviata (elimina_cartella_lezioni: {})...", should_delete_storage);

    // 1. Rilascia eventuale wake lock e termina immediatamente il backend Python se attivo
    if let Some(wls) = app.try_state::<WakeLockState>() {
        let _ = release_wake_lock(wls);
    }
    kill_backend(&app);
    std::thread::sleep(Duration::from_millis(500));

    // 2. Rileva e normalizza la cartella di archiviazione utente
    let home = get_home_dir();
    let mut effective_storage = keep_storage_path.filter(|s| !s.trim().is_empty());
    if effective_storage.is_none() {
        if let Some(ref h) = home {
            for cand in &[
                h.join("Library/Application Support/com.sbobinai.app/settings.json"),
                h.join(r"AppData\Local\SbobinAI\settings.json"),
                h.join(".sbobinai/settings.json"),
            ] {
                if cand.is_file() {
                    if let Ok(content) = std::fs::read_to_string(cand) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(sp) = json.get("storage_path").and_then(|v| v.as_str()) {
                                effective_storage = Some(sp.to_string());
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    // Se l'utente ha scelto l'eliminazione totale, NON preservare nulla
    let preserved_path: Option<PathBuf> = if should_delete_storage {
        None
    } else {
        effective_storage
            .as_deref()
            .map(|s| {
                if s.starts_with("~/") || s.starts_with("~\\") {
                    if let Some(ref h) = home {
                        return h.join(&s[2..]);
                    }
                }
                PathBuf::from(s)
            })
            .or_else(|| home.as_ref().map(|h| h.join("Documents").join("SbobinAI")))
    };

    if should_delete_storage {
        log::warn!("PULIZIA TOTALE: Nessun percorso preservato. La cartella lezioni e ogni traccia verranno rimosse.");
    } else {
        log::info!("Cartella dati utente PRESERVATA (whitelist protetta): {:?}", preserved_path);
    }

    let is_preserved = |p: &Path| -> bool {
        if let Some(ref pres) = preserved_path {
            if let (Ok(p_canon), Ok(pres_canon)) = (p.canonicalize(), pres.canonicalize()) {
                return p_canon == pres_canon 
                    || p_canon.starts_with(&pres_canon) 
                    || pres_canon.starts_with(&p_canon);
            }
            if p == pres.as_path() || p.starts_with(pres.as_path()) || pres.as_path().starts_with(p) {
                return true;
            }
        }
        false
    };

    let safe_remove_dir = |p: &Path| {
        if p.exists() && !is_preserved(p) {
            log::info!("Rimozione directory: {:?}", p);
            for attempt in 0..5 {
                match std::fs::remove_dir_all(p) {
                    Ok(_) => break,
                    Err(e) if attempt < 4 => {
                        log::warn!("Tentativo {} cancellazione {:?} fallito ({}), attendo rilascio lock...", attempt + 1, p, e);
                        std::thread::sleep(Duration::from_millis(300 * (attempt + 1) as u64));
                    }
                    Err(e) => {
                        log::error!("Impossibile rimuovere {:?} dopo ripetuti tentativi: {}", p, e);
                    }
                }
            }
        }
    };

    let safe_remove_file = |p: &Path| {
        if p.exists() && !is_preserved(p) {
            log::info!("Rimozione file: {:?}", p);
            let _ = std::fs::remove_file(p);
        }
    };

    let safe_delete_storage_directory = |target_path: &Path| {
        if !target_path.exists() {
            return;
        }

        // Controllo di sicurezza: non eliminare mai cartelle di sistema vitali
        if let Some(ref h) = home {
            let p_canon = target_path.canonicalize().unwrap_or_else(|_| target_path.to_path_buf());
            let h_canon = h.canonicalize().unwrap_or_else(|_| h.to_path_buf());
            let docs_canon = h.join("Documents").canonicalize().unwrap_or_else(|_| h.join("Documents"));
            let desk_canon = h.join("Desktop").canonicalize().unwrap_or_else(|_| h.join("Desktop"));
            let down_canon = h.join("Downloads").canonicalize().unwrap_or_else(|_| h.join("Downloads"));

            if p_canon == h_canon || p_canon == docs_canon || p_canon == desk_canon || p_canon == down_canon || p_canon.parent().is_none() {
                log::warn!("Il percorso storage coincide con una cartella radice o di sistema ({:?}). Elimino solo i dati interni di SbobinAI.", p_canon);
                safe_remove_dir(&target_path.join("audio"));
                safe_remove_dir(&target_path.join("sqlite"));
                safe_remove_dir(&target_path.join("lancedb"));
                safe_remove_dir(&target_path.join("uploads"));
                safe_remove_dir(&target_path.join("quaderno"));
                safe_remove_file(&target_path.join("settings.json"));
                safe_remove_file(&target_path.join("sbobinai.db"));
                safe_remove_file(&target_path.join("unicopilot.db"));
                return;
            }
        }

        log::info!("Eliminazione definitiva cartella storage SbobinAI: {:?}", target_path);
        safe_remove_dir(target_path);
    };

    // 3. Rimozione profonda dei modelli Hugging Face e delle cache AI di SbobinAI
    let mut hf_roots: Vec<PathBuf> = Vec::new();
    if let Ok(val) = std::env::var("HF_HUB_CACHE") {
        hf_roots.push(PathBuf::from(val));
    }
    if let Ok(val) = std::env::var("HF_HOME") {
        let p = PathBuf::from(val);
        hf_roots.push(p.join("hub"));
        hf_roots.push(p);
    }
    if let Some(ref h) = home {
        hf_roots.push(h.join(".cache").join("huggingface").join("hub"));
        hf_roots.push(h.join(".cache").join("huggingface"));
        hf_roots.push(h.join(".huggingface"));
        #[cfg(target_os = "macos")]
        {
            hf_roots.push(h.join("Library/Caches/huggingface"));
        }
        #[cfg(target_os = "windows")]
        {
            if let Ok(local_app) = std::env::var("LOCALAPPDATA") {
                let p = PathBuf::from(local_app);
                hf_roots.push(p.join(r"huggingface\hub"));
                hf_roots.push(p.join("huggingface"));
            }
            if let Ok(app_data) = std::env::var("APPDATA") {
                let p = PathBuf::from(app_data);
                hf_roots.push(p.join(r"huggingface\hub"));
                hf_roots.push(p.join("huggingface"));
            }
            hf_roots.push(h.join(r"AppData\Local\huggingface\hub"));
            hf_roots.push(h.join(r"AppData\Local\huggingface"));
            hf_roots.push(h.join(r"AppData\Roaming\huggingface"));
        }
    }

    if should_delete_storage {
        // Pulizia totale: elimina ogni singola directory di cache HuggingFace
        for hf_dir in &hf_roots {
            safe_remove_dir(hf_dir);
        }

        // Cache PyTorch, Whisper, MLX, uv e pip
        if let Some(ref h) = home {
            safe_remove_dir(&h.join(".cache/mlx"));
            safe_remove_dir(&h.join(".cache/torch"));
            safe_remove_dir(&h.join(".cache/whisper"));
            safe_remove_dir(&h.join(".cache/uv"));
            safe_remove_dir(&h.join(".local/share/uv"));
            #[cfg(target_os = "macos")]
            {
                safe_remove_dir(&h.join("Library/Caches/pip"));
                safe_remove_dir(&h.join("Library/Caches/uv"));
            }
            #[cfg(target_os = "windows")]
            {
                safe_remove_dir(&h.join(r"AppData\Local\torch"));
                safe_remove_dir(&h.join(r"AppData\Local\whisper"));
                safe_remove_dir(&h.join(r"AppData\Local\pip"));
                safe_remove_dir(&h.join(r"AppData\Local\uv"));
                safe_remove_dir(&h.join(r"AppData\Roaming\uv"));
            }
        }
    } else {
        // Se preserva la cartella lezioni, rimuove comunque tutti i modelli AI scaricati da SbobinAI
        let sbobinai_prefixes = [
            "models--mlx-community--whisper",
            "models--dropbox-dash--faster-whisper",
            "models--Systran--faster-whisper",
            "models--mlx-community--Qwen",
            "models--unsloth--Qwen",
            "models--unsloth--gemma",
            "models--mlx-community--bge",
            "models--mlx-community--multilingual",
            "models--CompendiumLabs--bge",
            "models--Qwen--Qwen3-Embedding",
            "models--Qwen--Qwen",
        ];

        for hf_cache in &hf_roots {
            if hf_cache.is_dir() {
                if let Ok(entries) = std::fs::read_dir(hf_cache) {
                    for entry in entries.flatten() {
                        let name = entry.file_name().to_string_lossy().to_string();
                        if sbobinai_prefixes.iter().any(|prefix| name.starts_with(prefix)) {
                            safe_remove_dir(&entry.path());
                        }
                    }
                }

                let locks_dir = hf_cache.join(".locks");
                if locks_dir.is_dir() {
                    if let Ok(entries) = std::fs::read_dir(&locks_dir) {
                        for entry in entries.flatten() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            if sbobinai_prefixes.iter().any(|prefix| name.starts_with(prefix)) {
                                safe_remove_dir(&entry.path());
                                safe_remove_file(&entry.path());
                            }
                        }
                    }
                }
            }
        }
    }

    // 4. Rimozione dei dati applicazione, venv, backend e settings
    if let Some(ref h) = home {
        #[cfg(target_os = "macos")]
        {
            safe_remove_dir(&h.join("Library/Application Support/com.sbobinai.app"));
            safe_remove_dir(&h.join("Library/Application Support/SbobinAI"));
            safe_remove_dir(&h.join("Library/Application Support/unicopilot"));
            safe_remove_dir(&h.join("Library/Caches/com.sbobinai.app"));
            safe_remove_dir(&h.join("Library/Caches/SbobinAI"));
            safe_remove_dir(&h.join("Library/Caches/unicopilot"));
            safe_remove_dir(&h.join("Library/WebKit/com.sbobinai.app"));
            safe_remove_dir(&h.join("Library/WebKit/SbobinAI"));
            safe_remove_dir(&h.join("Library/Logs/SbobinAI"));
            safe_remove_dir(&h.join("Library/Logs/com.sbobinai.app"));
            safe_remove_dir(&h.join("Library/Saved Application State/com.sbobinai.app.savedState"));
            safe_remove_dir(&h.join("Library/HTTPStorages/com.sbobinai.app"));
            safe_remove_file(&h.join("Library/Preferences/com.sbobinai.app.plist"));
            safe_remove_dir(&h.join(".sbobinai"));
            safe_remove_dir(&h.join(".unicopilot"));
        }

        #[cfg(target_os = "windows")]
        {
            if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
                let lad = std::path::PathBuf::from(&local_app_data);
                safe_remove_dir(&lad.join(r"SbobinAI\cuda_runtime"));
                safe_remove_dir(&lad.join(r"SbobinAI\cuda_runtime.staging"));
                safe_remove_dir(&lad.join("SbobinAI"));
                safe_remove_dir(&lad.join("com.sbobinai.app"));
                safe_remove_dir(&lad.join("unicopilot"));
                safe_remove_dir(&lad.join("EBWebView"));
                safe_remove_dir(&lad.join(r"Programs\SbobinAI"));
            }
            safe_remove_dir(&h.join(r"AppData\Local\SbobinAI\cuda_runtime"));
            safe_remove_dir(&h.join(r"AppData\Local\SbobinAI\cuda_runtime.staging"));
            safe_remove_dir(&h.join(r"AppData\Local\SbobinAI"));
            safe_remove_dir(&h.join(r"AppData\Local\com.sbobinai.app"));
            safe_remove_dir(&h.join(r"AppData\Local\unicopilot"));
            safe_remove_dir(&h.join(r"AppData\Local\EBWebView"));
            safe_remove_dir(&h.join(r"AppData\Roaming\com.sbobinai.app"));
            safe_remove_dir(&h.join(r"AppData\Roaming\SbobinAI"));
            safe_remove_dir(&h.join(r"AppData\Roaming\unicopilot"));
            safe_remove_dir(&h.join(r"AppData\Local\Programs\SbobinAI"));
            safe_remove_file(&h.join(r"Desktop\SbobinAI.lnk"));
            safe_remove_file(&h.join(r"AppData\Roaming\Microsoft\Windows\Start Menu\Programs\SbobinAI.lnk"));
            safe_remove_dir(&h.join(".sbobinai"));
            safe_remove_dir(&h.join(".unicopilot"));
        }

        #[cfg(all(unix, not(target_os = "macos")))]
        {
            safe_remove_dir(&h.join(".sbobinai"));
            safe_remove_dir(&h.join(".unicopilot"));
            safe_remove_dir(&h.join(".config/com.sbobinai.app"));
            safe_remove_dir(&h.join(".config/SbobinAI"));
            safe_remove_dir(&h.join(".local/share/com.sbobinai.app"));
            safe_remove_dir(&h.join(".local/share/SbobinAI"));
            safe_remove_dir(&h.join(".cache/com.sbobinai.app"));
            safe_remove_dir(&h.join(".cache/SbobinAI"));
        }
    }

    // 5. Pulizia file temporanei
    let temp_dir = std::env::temp_dir();
    if let Ok(entries) = std::fs::read_dir(&temp_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("sbobinai_")
                || name.starts_with("oral_answer_")
                || name.starts_with("unicopilot_")
                || name.starts_with("huggingface")
                || name.starts_with("torch_")
                || name == "com.sbobinai.app"
            {
                let _ = std::fs::remove_file(entry.path());
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    // 5b. Se l'utente ha scelto l'eliminazione totale, elimina la cartella delle lezioni e dei dati personali
    if should_delete_storage {
        if let Some(ref raw_str) = effective_storage {
            let resolved = if raw_str.starts_with("~/") || raw_str.starts_with("~\\") {
                home.as_ref().map(|h| h.join(&raw_str[2..]))
            } else {
                Some(PathBuf::from(raw_str))
            };
            if let Some(p) = resolved {
                safe_delete_storage_directory(&p);
            }
        }

        if let Some(ref h) = home {
            safe_delete_storage_directory(&h.join("Documents").join("SbobinAI"));
            safe_delete_storage_directory(&h.join("Documents").join("UniCoPilot"));
            #[cfg(target_os = "windows")]
            {
                safe_delete_storage_directory(&h.join(r"OneDrive\Documents\SbobinAI"));
                safe_delete_storage_directory(&h.join(r"OneDrive\Documents\UniCoPilot"));
                safe_delete_storage_directory(&h.join(r"OneDrive\Documenti\SbobinAI"));
                safe_delete_storage_directory(&h.join(r"Documenti\SbobinAI"));
            }
            safe_delete_storage_directory(&h.join(".sbobinai"));
            safe_delete_storage_directory(&h.join(".unicopilot"));
        }
    }

    // 6. Autodistruzione del pacchetto applicativo se in produzione
    if let Ok(exe_path) = std::env::current_exe() {
        let is_dev_build = exe_path
            .ancestors()
            .any(|a| a.join(".git").exists() || a.ends_with("target/debug") || a.ends_with("target/release"));

        if !is_dev_build {
            #[cfg(target_os = "macos")]
            {
                let app_bundle = exe_path
                    .ancestors()
                    .find(|a| a.extension().map_or(false, |ext| ext == "app"));

                if let Some(bundle) = app_bundle {
                    let bundle_str = bundle.to_string_lossy().to_string().replace('\'', "'\\''");
                    let home_str = home.as_ref().map(|h| h.to_string_lossy().to_string()).unwrap_or_default().replace('\'', "'\\''");
                    
                    let mut storage_clean_cmd = String::new();
                    if should_delete_storage {
                        if let Some(ref raw_str) = effective_storage {
                            let resolved = if raw_str.starts_with("~/") || raw_str.starts_with("~\\") {
                                format!("{}/{}", home_str, &raw_str[2..])
                            } else {
                                raw_str.clone()
                            };
                            if !resolved.is_empty() && resolved != home_str && resolved != format!("{}/Documents", home_str) && resolved != format!("{}/Desktop", home_str) {
                                storage_clean_cmd.push_str(&format!("rm -rf '{}' 2>/dev/null; ", resolved.replace('\'', "'\\''")));
                            }
                        }
                        storage_clean_cmd.push_str(&format!("rm -rf '{}/Documents/SbobinAI' '{}/Documents/UniCoPilot' 2>/dev/null; ", home_str, home_str));
                        storage_clean_cmd.push_str(&format!("rm -rf '{0}/.cache/huggingface' '{0}/.huggingface' '{0}/Library/Caches/huggingface' '{0}/.cache/torch' '{0}/.cache/mlx' '{0}/.cache/whisper' '{0}/.cache/uv' '{0}/.local/share/uv' '{0}/Library/Caches/pip' '{0}/Library/Caches/uv' 2>/dev/null; ", home_str));
                    }

                    log::info!("Avvio script detached di rimozione per bundle: {}", bundle_str);
                    let _ = Command::new("/bin/sh")
                        .args([
                            "-c",
                            &format!(
                                "sleep 1.2 && rm -rf '{bundle}' '{home}/Library/Application Support/com.sbobinai.app' '{home}/Library/Application Support/SbobinAI' '{home}/Library/Application Support/unicopilot' '{home}/Library/Caches/com.sbobinai.app' '{home}/Library/Caches/SbobinAI' '{home}/Library/Caches/unicopilot' '{home}/Library/WebKit/com.sbobinai.app' '{home}/Library/WebKit/SbobinAI' '{home}/Library/Logs/SbobinAI' '{home}/Library/Logs/com.sbobinai.app' '{home}/Library/Saved Application State/com.sbobinai.app.savedState' '{home}/Library/HTTPStorages/com.sbobinai.app' '{home}/Library/Preferences/com.sbobinai.app.plist' '{home}/.sbobinai' '{home}/.unicopilot' 2>/dev/null; {storage_clean_cmd}defaults delete com.sbobinai.app 2>/dev/null || true; killall Dock 2>/dev/null || true",
                                bundle = bundle_str,
                                home = home_str,
                                storage_clean_cmd = storage_clean_cmd
                            ),
                        ])
                        .stdin(Stdio::null())
                        .stdout(Stdio::null())
                        .stderr(Stdio::null())
                        .spawn();
                }
            }

            #[cfg(target_os = "windows")]
            {
                let exe_dir = exe_path.parent().map(|p| p.to_string_lossy().to_string()).unwrap_or_default().replace('\'', "''");
                let uninstaller = exe_path.parent().map(|p| p.join("Uninstall SbobinAI.exe"));
                let has_uninstaller = uninstaller.as_ref().map(|u| u.is_file()).unwrap_or(false);
                
                if let Some(uninst) = uninstaller {
                    if has_uninstaller {
                        let _ = silent_command(&uninst)
                            .arg("/S")
                            .spawn();
                    }
                }

                let mut storage_ps_cmd = String::new();
                if should_delete_storage {
                    if let Some(ref raw_str) = effective_storage {
                        let resolved = if raw_str.starts_with("~/") || raw_str.starts_with("~\\") {
                            home.as_ref().map(|h| h.join(&raw_str[2..]).to_string_lossy().to_string()).unwrap_or_default()
                        } else {
                            raw_str.clone()
                        };
                        if !resolved.is_empty() {
                            storage_ps_cmd.push_str(&format!("Remove-Item -Recurse -Force '{}' -ErrorAction SilentlyContinue; ", resolved.replace('\'', "''")));
                        }
                    }
                    if let Some(ref h) = home {
                        let default_doc = h.join("Documents").join("SbobinAI").to_string_lossy().to_string().replace('\'', "''");
                        storage_ps_cmd.push_str(&format!("Remove-Item -Recurse -Force '{}' -ErrorAction SilentlyContinue; ", default_doc));
                        let unist_doc = h.join("Documents").join("UniCoPilot").to_string_lossy().to_string().replace('\'', "''");
                        storage_ps_cmd.push_str(&format!("Remove-Item -Recurse -Force '{}' -ErrorAction SilentlyContinue; ", unist_doc));
                    }
                    storage_ps_cmd.push_str(
                        "Remove-Item -Recurse -Force \"$HOME\\.cache\\huggingface\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$HOME\\.huggingface\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\huggingface\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:APPDATA\\huggingface\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$HOME\\.cache\\torch\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\torch\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$HOME\\.cache\\whisper\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\whisper\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$HOME\\.cache\\uv\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\uv\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:APPDATA\\uv\" -ErrorAction SilentlyContinue; \
                         Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\pip\" -ErrorAction SilentlyContinue; \
                         $docs = [Environment]::GetFolderPath('MyDocuments'); if ($docs) { Remove-Item -Recurse -Force \"$docs\\SbobinAI\" -ErrorAction SilentlyContinue; Remove-Item -Recurse -Force \"$docs\\UniCoPilot\" -ErrorAction SilentlyContinue; }; "
                    );
                }

                let exe_cleanup = if has_uninstaller {
                    String::new()
                } else {
                    format!("Remove-Item -Recurse -Force '{}' -ErrorAction SilentlyContinue; ", exe_dir)
                };

                let _ = silent_command("powershell")
                    .args([
                        "-NoProfile",
                        "-WindowStyle",
                        "Hidden",
                        "-Command",
                        &format!(
                            "Start-Sleep -Seconds 2; \
                             Get-Process SbobinAI -ErrorAction SilentlyContinue | Stop-Process -Force; \
                             Get-Process msedgewebview2 -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '*com.sbobinai.app*' }} | Stop-Process -Force; \
                             Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object {{ $_.Path -like '*SbobinAI*' }} | Stop-Process -Force; \
                             {}Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\SbobinAI\\cuda_runtime\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\SbobinAI\\cuda_runtime.staging\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\SbobinAI\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\com.sbobinai.app\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\unicopilot\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\EBWebView\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:APPDATA\\com.sbobinai.app\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:APPDATA\\SbobinAI\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:APPDATA\\unicopilot\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$HOME\\.sbobinai\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$HOME\\.unicopilot\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force \"$env:LOCALAPPDATA\\Programs\\SbobinAI\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Force \"$env:USERPROFILE\\Desktop\\SbobinAI.lnk\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Force \"$env:APPDATA\\Microsoft\\Windows\\Start Menu\\Programs\\SbobinAI.lnk\" -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force 'HKCU:\\Software\\SbobinAI' -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force 'HKCU:\\Software\\com.sbobinai.app' -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force 'HKCU:\\Software\\unicopilot' -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\com.sbobinai.app' -ErrorAction SilentlyContinue; \
                             Remove-Item -Recurse -Force 'HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\SbobinAI' -ErrorAction SilentlyContinue; {}",
                            exe_cleanup,
                            storage_ps_cmd
                        ),
                    ])
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn();
            }
        } else {
            log::info!("Ambiente di sviluppo rilevato: saltata cancellazione binario.");
        }
    }

    // 7. Chiusura del processo
    let app_handle_for_exit = app.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(600));
        app_handle_for_exit.exit(0);
    });

    Ok(())
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
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    let backend_state = BackendState(backend_status);
    let backend_proc = BackendProcess(backend_proc_arc.clone());

    let state_for_setup = backend_state.clone();
    let proc_for_setup = BackendProcess(backend_proc_arc);
    let shutdown_for_setup = shutdown_flag.clone();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .manage(backend_state)
        .manage(backend_proc)
        .manage(AppShutdown(shutdown_flag))
        .manage(WakeLockState::default())
        .invoke_handler(tauri::generate_handler![
            get_backend_status,
            stop_backend_for_update,
            uninstall_sbobinai,
            acquire_wake_lock,
            release_wake_lock
        ])
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
                run_backend_bootstrap(app_handle, state_for_setup, proc_for_setup, shutdown_for_setup);
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
            event: WindowEvent::CloseRequested { .. },
            ..
        } if label == "main" => {
            if let Some(wls) = app.try_state::<WakeLockState>() {
                let _ = release_wake_lock(wls);
            }
            kill_backend(app);
            app.exit(0);
        }
        RunEvent::WindowEvent {
            label,
            event: WindowEvent::Destroyed,
            ..
        } if label == "main" => {
            if let Some(wls) = app.try_state::<WakeLockState>() {
                let _ = release_wake_lock(wls);
            }
            kill_backend(app);
            app.exit(0);
        }
        RunEvent::Exit => {
            if let Some(wls) = app.try_state::<WakeLockState>() {
                let _ = release_wake_lock(wls);
            }
            kill_backend(app);
        }
        _ => {}
    });
}
