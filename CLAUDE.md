# SbobinAI — note per chi continua su macOS

Assistente universitario locale: registra le lezioni, le trascrive, genera il quaderno di studio, quiz e simulazione d'esame orale.
Backend FastAPI (`backend/`) + frontend React/TypeScript dentro Tauri 2 (`frontend/`). Interfaccia e messaggi sono in italiano: mantienili tali.

## Perché questo file
Il lavoro recente (audit bug, supporto Windows, test) è stato fatto su un PC **Windows**, dove non è possibile compilare l'app Tauri
(Smart App Control blocca le DLL delle proc-macro di `rustc`). Su Mac **non è stato eseguito nulla**: tocca a te verificare i punti qui sotto.

## Da fare per primo sul Mac
1. `git pull`, poi backend:
   ```bash
   cd backend && python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt        # sul Mac arm64 installa lo stack MLX
   python -m pytest -q                    # deve dare 48 passed (nessun modello reale richiesto)
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```
2. Frontend + app desktop: `cd frontend && npm install && npm run tauri dev` (serve Rust: `rustup`).
3. Controlla che **MLX sia ancora quello di prima** (vedi sotto) e che l'app Tauri compili e si apra.

## Cosa NON è stato verificato (verificalo tu)
- **Percorso MLX**: `transcriber.py`, `llm_engine.py`, `rag.py` ora scelgono lo stack per piattaforma (`platform_utils.py`, `ai_models.py`).
  Su macOS arm64 devono usare `mlx-whisper`, `mlx-lm`, `mlx-embeddings` con gli stessi parametri di prima. Provato solo con test e stub, mai con MLX reale.
  Prova: scarica i modelli dalle Impostazioni, registra 1 minuto in live, genera il capitolo, fai una domanda in chat, un quiz.
- **Build Tauri su macOS**: la config è stata divisa. `frontend/src-tauri/tauri.conf.json` è neutra (finestra opaca),
  `tauri.macos.conf.json` rimette `transparent`, `hiddenTitle`, `titleBarStyle: Overlay`. In `Cargo.toml` c'è la feature `macos-private-api`
  e `macOSPrivateApi: true` nella config base. Se la finestra "Liquid Glass" (vibrancy in `src-tauri/src/lib.rs`) non è più trasparente, parti da qui.
- **Permessi microfono**: `frontend/src-tauri/Info.plist` (NSMicrophoneUsageDescription, NSSpeechRecognitionUsageDescription). Verifica nell'app **bundle** (`npm run tauri build`), non solo in dev.
- **CSP** in `tauri.conf.json`: `media-src` per il player audio, `connect-src` per http/ws su `localhost`/`127.0.0.1` porte 8000 e 8765. Verifica che audio e WebSocket funzionino nell'app.

## Architettura in breve
- `backend/main.py`: app FastAPI, WebSocket `/ws/live-lecture/{subject_id}` (registrazione live), endpoint di riepilogo/chat/reindex.
  La chiusura di una sessione passa **sempre** da `_finalize_session` (idempotente, chiamata anche in `finally` del WebSocket): non salvare la lezione altrove.
- `backend/live_session.py`: sessione live in RAM (una per materia). `end_session(subject_id, session)` chiude solo la sessione indicata.
- `backend/audio_engine.py`: microfono + VAD, buffer su file temporaneo (spill su disco oltre 64 MB), export WAV.
- `backend/llm_engine.py`: generazione. API da usare: `generate_text(...)` (bloccante), `stream_chat_response`, `generate_lecture_chapter` (lancia `LLMGenerationError`/`InterruptedError`),
  stop con `cancel_epoch()`/`is_cancelled()`. Non usare più `_cancel_generation_event` (deprecato) né assegnare `_is_generating`/`_current_task`.
- `backend/transcriber.py`: `Transcriber` (live) e `transcribe_file(path, language, initial_prompt) -> (testo, durata)` (file).
- `backend/rag.py`: LanceDB, embedding a 1024 dimensioni (non cambiare: lo schema è fisso), ricerca ibrida.
- `backend/config.py`: percorsi e `settings.json`. Per modificare le impostazioni usa `update_settings()` (atomico, con lock), mai load→modifica→save a mano.
  `SBOBINAI_SETTINGS_FILE` sposta il file (usato dai test).
- `backend/routers/`: subjects, lectures, settings, models, study.
- Gli ID dei modelli sono **uguali su tutte le piattaforme** (`large-v3-turbo-q4`, `mlx-community/Qwen3.5-4B-MLX-4bit`, `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ`);
  `ai_models.py` li mappa al repo giusto per piattaforma.

## Regole da non rompere
- Il server ascolta **solo** su `127.0.0.1` e il WebSocket controlla l'`Origin` (localhost, `tauri://localhost`, `http(s)://tauri.localhost`). Non rimettere `0.0.0.0`.
- Nome file degli upload: passa sempre da `_safe_filename` (path traversal).
- Handler pesanti (embedding, PDF, Whisper, LanceDB) fuori dall'event loop: handler `def` sincroni oppure `asyncio.to_thread`.
- Il cleaner all'avvio non cancella nulla se il DB non ha materie: non togliere quella salvaguardia.
- Filtri LanceDB costruiti a stringa: raddoppia sempre gli apostrofi.

## Test
`cd backend && python -m pytest -q` — usa storage e `settings.json` temporanei, WebSocket live in mock mode, nessun modello scaricato.
Se aggiungi codice che tocca LLM/Whisper/RAG, stub-alo nei test come in `tests/conftest.py` e `tests/test_study.py`.

## Porte in sviluppo
Default 8000 (backend) e 5173 (Vite). Se la 8000 è occupata (es. Docker) usa un'altra porta e passa
`VITE_API_URL=http://localhost:<porta>/api/v1` e `VITE_WS_URL=ws://localhost:<porta>`; la CSP ammette già 8000 e 8765.

## Aggiornamento: sidecar, updater, CI (NON compilati)
Aggiunti da Windows senza poter eseguire `cargo`: **il primo `cargo build` sul Mac è il vero test** di `frontend/src-tauri/src/lib.rs`
(avvio/chiusura automatica del backend Python sulla 8000, Mica/Acrylic su Windows, vibrancy su macOS, scorciatoia globale `CommandOrControl+Shift+B` → evento `global-bookmark`,
plugin updater/process). Se la mappa `bundle.resources` in `tauri.conf.json` viene rifiutata, passa a voci per singolo file.
- Requirements separati: `backend/requirements.txt` = solo macOS/MLX, `backend/requirements-windows.txt` = Windows/Linux. Guida: `WINDOWS_BUILD_GUIDE.md`.
- Release: `.github/workflows/release.yml` (tag `v*`). Servono i segreti GitHub `TAURI_SIGNING_PRIVATE_KEY` e `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
  La chiave pubblica è già in `tauri.conf.json`; la privata NON è nel repo (chi l'ha generata la tiene fuori dal repo).
- Sito: `NETLIFY_INTEGRATION.md` (bottone download automatico per OS).
