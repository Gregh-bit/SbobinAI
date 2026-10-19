use std::net::{SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;

use tauri::{Emitter, Manager, RunEvent, WindowEvent};

#[cfg(target_os = "macos")]
use window_vibrancy::{NSVisualEffectMaterial, apply_vibrancy};
#[cfg(target_os = "windows")]
use window_vibrancy::{apply_acrylic, apply_mica};

/// Python backend process spawned by us (None if already running or spawn failed).
struct BackendProcess(Mutex<Option<Child>>);

fn backend_is_up() -> bool {
    let addr: SocketAddr = ([127, 0, 0, 1], 8000).into();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn has_main(dir: &Path) -> bool {
    dir.join("main.py").is_file()
}

fn resolve_backend_dir(app: &tauri::App) -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SBOBINAI_BACKEND_DIR") {
        let p = PathBuf::from(p);
        if has_main(&p) {
            return Some(p);
        }
    }
    if let Ok(res) = app.path().resource_dir() {
        let p = res.join("backend");
        if has_main(&p) {
            return Some(p);
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

fn resolve_python(backend_dir: &Path) -> PathBuf {
    let rel: &[&str] = if cfg!(windows) {
        &["venv/Scripts/python.exe", ".venv/Scripts/python.exe"]
    } else {
        &["venv/bin/python", ".venv/bin/python"]
    };
    for r in rel {
        let p = backend_dir.join(r);
        if p.is_file() {
            return p;
        }
    }
    PathBuf::from(if cfg!(windows) { "python" } else { "python3" })
}

fn spawn_backend(app: &tauri::App) -> Option<Child> {
    if backend_is_up() {
        log::info!("backend already running on 127.0.0.1:8000, not spawning");
        return None;
    }
    let dir = match resolve_backend_dir(app) {
        Some(d) => d,
        None => {
            log::warn!("backend directory not found, backend not started");
            return None;
        }
    };
    let python = resolve_python(&dir);
    let mut cmd = Command::new(&python);
    cmd.args([
        "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000",
    ])
    .current_dir(&dir)
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    match cmd.spawn() {
        Ok(child) => {
            log::info!("backend spawned (pid {}) from {:?}", child.id(), dir);
            Some(child)
        }
        Err(e) => {
            log::warn!("failed to spawn backend with {:?}: {}", python, e);
            None
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
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_process::init())
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            // Python backend
            let child = spawn_backend(app);
            app.manage(BackendProcess(Mutex::new(child)));

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
