# SbobinAI — Guida alla build su Windows

Per chi compila e prova la versione Windows (Windows 10/11 x64, PowerShell).

## 1. Prerequisiti
| Cosa | Come |
|---|---|
| Python 3.12 (3.14 funziona) | `winget install Python.Python.3.12` |
| Node.js 20+ | `winget install OpenJS.NodeJS.LTS` |
| Rust | `winget install Rustlang.Rustup` poi `rustup default stable` |
| Build Tools C++ (obbligatori per Tauri) | `winget install Microsoft.VisualStudio.2022.BuildTools --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` |
| WebView2 | già incluso in Windows 11 |

> **Smart App Control**: se è attivo (Sicurezza di Windows → Controllo app e browser) blocca `rustc` e alcune DLL (es. pyarrow 26).
> Sintomo: `can't find crate for 'tauri_macros'` oppure `DLL load failed ... criterio di controllo dell'applicazione`.
> Va disattivato (irreversibile senza reinstallare Windows), oppure si compila su un'altra macchina/VM o con la CI di GitHub.

## 2. Backend Python
```powershell
cd backend
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-windows.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m pytest -q          # 48 test, nessun modello richiesto
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```
Stack Windows: **faster-whisper** (trascrizione) e **llama-cpp-python** (LLM Qwen3.5 GGUF + embedding Qwen3-Embedding GGUF a 1024 dimensioni).
I modelli si scaricano dalle Impostazioni dell'app (~6 GB in totale). Gli ID dei modelli sono identici a quelli macOS: `ai_models.py` li traduce.
Il codice MLX non viene mai importato su Windows e le librerie Windows non vengono mai importate su Mac (import lazy in `platform_utils.py`).

### GPU NVIDIA (facoltativa)
- Whisper: installa CUDA 12 + cuDNN 9; senza, usa la CPU in automatico.
- LLM: non esiste una wheel CUDA recente, va compilato: `$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install llama-cpp-python --no-cache-dir --force-reinstall` (richiede il CUDA Toolkit).
- Forzare il backend: `$env:SBOBINAI_AI_BACKEND="portable"` (oppure `mlx`). Microfono: `$env:SBOBINAI_INPUT_DEVICE`.

## 3. App desktop (Tauri)
```powershell
cd frontend
npm install
npm run tauri dev       # sviluppo
npm run tauri build     # installer .exe (NSIS) e .msi in src-tauri\target\release\bundle
```
L'app avvia da sola il backend Python se la porta 8000 è libera (usa `backend\venv\Scripts\python.exe`) e lo chiude all'uscita.
Se la 8000 è occupata (es. Docker) avvia tu il backend su un'altra porta e imposta
`$env:VITE_API_URL="http://localhost:<porta>/api/v1"; $env:VITE_WS_URL="ws://localhost:<porta>"`.

## 4. Cosa controllare
- Registrazione live: microfono (Privacy → Microfono), trascrizione in italiano, pausa/ripresa, stop.
- Scorciatoia globale `Ctrl+Shift+B`: salva un segnalibro anche con l'app in secondo piano.
- Finestra con effetto Mica/Acrylic (Windows 11).
- Aggiornamenti dalle Impostazioni (serve una Release GitHub pubblicata).

## 5. Release
Push di un tag `vX.Y.Z` → GitHub Actions (`.github/workflows/release.yml`) produce `.dmg` e `.exe/.msi` firmati e `latest.json`.
Segreti richiesti nella repo: `TAURI_SIGNING_PRIVATE_KEY` e `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` (vuota se la chiave non ha password).
