# SbobinAI 🎓

> **L'assistente universitario intelligente, 100% locale e privato, ottimizzato per Apple Silicon.**

SbobinAI trasforma le registrazioni delle lezioni universitarie e i materiali didattici in un quaderno di studio interattivo, quiz d'esame e simulazioni orali, eseguendo modelli di intelligenza artificiale all'avanguardia **interamente in locale su Apple Silicon Metal**, senza inviare alcun dato all'esterno e senza consumare batteria o memoria swap.

---

## ✨ Funzionalità Chiave

- 🎙️ **Registrazione Live & Trascrizione Streaming**: Trascrizione in tempo reale con `mlx-whisper` (Large-v3-Turbo quantizzato a 4-bit) con audio buffering in RAM a zero disk I/O.
- 🔖 **Segnalibri Esame & Player Sincronizzato**: Salva i momenti chiave indicati dal docente durante la lezione e riascoltali con un clic saltando al secondo esatto (`click-to-play`).
- 📖 **Quaderno di Studio Digitale**: Generazione automatica di schemi concettuali e capitoli approfonditi in Markdown tramite `Qwen 3.5 4B MLX`. Esportazione con un clic in `.md`.
- 📚 **Ingestione Reale dei Materiali**: Caricamento di slide, PDF (estrazione ultra-rapida con PyMuPDF) e audio esterni con indicizzazione semantica ibrida immediata su `LanceDB`.
- 🧠 **Quiz & Simulazione Esame Orale**:
  - **Quiz a Scelta Multipla**: 5 domande d'esame contestualizzate dalle lezioni con correzione e spiegazione.
  - **Interrogazione Vocale**: Parla al microfono, l'IA trascrive la risposta e la valuta confrontandola con gli appunti (voto in trentesimi, punti di forza, lacune e formulazione ideale).
- 🧹 **Garbage Collector & Storage Optimizer**: Monitoraggio dello spazio su disco, eliminazione fisica a cascata di audio/materiali e compattazione automatica (`VACUUM` SQLite e `optimize` LanceDB).
- 🎨 **Interfaccia Nativa macOS Liquid Glass**: Applicazione desktop nativa realizzata con Tauri 2.0 (Rust) e React 18, fluida, reattiva e con supporto Dark Mode integrato.

---

## 🛠️ Stack Tecnologico

### Frontend
- **Framework Desktop**: [Tauri 2.0](https://tauri.app/) (Rust)
- **UI / Core**: React 18, TypeScript, Vite
- **Stili & Animazioni**: Tailwind CSS, Framer Motion (Apple Liquid Glass design system)
- **Icone & Markdown**: Lucide React, Marked

### Backend
- **Server API**: FastAPI, Uvicorn, WebSockets
- **Database Relazionale**: SQLite con SQLAlchemy ORM
- **Database Vettoriale & RAG**: LanceDB, PyArrow
- **Motore Vocale (STT)**: `mlx-whisper` (Whisper Large-v3-Turbo Q4 Metal)
- **Motore LLM**: `mlx-lm` (Qwen 3.5 4B 4-bit Metal)
- **Motore Embeddings**: `mlx-embeddings` (Qwen3-Embedding-0.6B-4bit-DWQ, 1024 dim)
- **Elaborazione Documenti**: PyMuPDF (`pymupdf`) in C

---

## 🚀 Requisiti di Sistema

- **macOS**: macOS Sonoma (14.0+) o successivo
- **Hardware**: Mac con processore Apple Silicon (M1, M2, M3, M4 con memoria unificata ≥ 8 GB)
- **Windows**: Windows 10/11 x64, vedi la sezione [Windows](#-windows-x64)
- **Linguaggi**: Python 3.10+ (consigliato 3.12), Node.js 18+, Rust (tramite `rustup`)

---

## 💻 Installazione & Avvio Rapido

### 1. Clona il repository
```bash
git clone https://github.com/Gregh-bit/SbobinAI.git
cd SbobinAI
```

### 2. Configura il Backend (Python)
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### 3. Avvia l'Applicazione Desktop (Tauri + React)
In un altro terminale:
```bash
cd frontend
npm install
npm run tauri dev
```

---

## 🪟 Windows (x64)

SbobinAI funziona anche su Windows 10/11 x64. Il backend sceglie automaticamente lo stack IA in base alla piattaforma (`backend/platform_utils.py`): su Apple Silicon resta MLX, su Windows usa uno stack portabile con GPU NVIDIA se presente e fallback automatico su CPU.

| Ruolo | macOS Apple Silicon | Windows / Linux |
|---|---|---|
| Trascrizione | `mlx-whisper` · Whisper Large-v3-Turbo Q4 | `faster-whisper` (CTranslate2) · Large-v3-Turbo, CUDA float16 o CPU int8 |
| LLM | `mlx-lm` · Qwen 3.5 4B MLX 4-bit | `llama-cpp-python` · Qwen 3.5 4B GGUF Q4_K_M (offload GPU se la build lo supporta) |
| Embeddings | `mlx-embeddings` · Qwen3-Embedding-0.6B 4-bit | `llama-cpp-python` · Qwen3-Embedding-0.6B GGUF Q8_0 |

Gli ID dei modelli nelle Impostazioni sono gli stessi su entrambe le piattaforme, e lo schema LanceDB (vettori a 1024 dimensioni) è identico. I modelli si scaricano dall'app (Impostazioni → Modelli) nella cache HuggingFace (`%USERPROFILE%\.cache\huggingface\hub`, oppure il percorso in `HF_HOME`).

### Requisiti
- Windows 10/11 x64, ≥ 16 GB di RAM consigliati (8 GB bastano solo in CPU con un modello alla volta)
- Python 3.12 x64 consigliato (testato anche con 3.14), Node.js 18+, Rust tramite `rustup` (toolchain MSVC) e WebView2 (già presente su Windows 11)
- Opzionale, GPU NVIDIA: driver recente, CUDA 12 e cuDNN 9 per `faster-whisper`; CUDA Toolkit 12 e Visual Studio Build Tools (C++) per compilare `llama-cpp-python` con CUDA
- ffmpeg non serve: l'audio viene decodificato con PyAV

### Backend (PowerShell)
```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# Wheel CPU precompilata di llama-cpp-python (nessun compilatore richiesto)
pip install -r requirements-windows.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Se PowerShell blocca `Activate.ps1`, esegui una volta `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

### Accelerazione GPU NVIDIA (opzionale)
```powershell
# Trascrizione: librerie CUDA per CTranslate2 (rilevate automaticamente all'avvio)
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

# LLM + embeddings: ricompila llama-cpp-python con CUDA (da "Developer PowerShell for VS")
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
pip install --force-reinstall --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python>=0.3.36"
```
Senza questi passaggi tutto funziona comunque su CPU. Se l'inizializzazione CUDA fallisce, Whisper torna automaticamente su CPU int8.

### App desktop
```powershell
cd frontend
npm install
npm run tauri dev      # sviluppo
npm run tauri build    # installer .msi / .exe (NSIS) in src-tauri\target\release\bundle
```

### Variabili d'ambiente utili
- `SBOBINAI_INPUT_DEVICE`: microfono da usare (indice o parte del nome, es. `"USB"`); altrimenti si usa quello predefinito di Windows
- `SBOBINAI_AI_BACKEND`: forza `mlx` o `portable`
- `SBOBINAI_LLM_CTX`: contesto LLM in token (default `8192`)

### Risoluzione problemi
- **"Un criterio di controllo dell'applicazione ha bloccato il file" / "DLL load failed"** (spesso `pyarrow` o `pymupdf`): è Windows Smart App Control che blocca DLL non firmate. Installa una versione diversa del pacchetto (es. `pip install "pyarrow<26"`), oppure disattiva Smart App Control in Sicurezza di Windows → Controllo app e browser.
- **Microfono non rilevato**: abilita l'accesso in Impostazioni → Privacy e sicurezza → Microfono ("Consenti alle app desktop di accedere al microfono"). Se il dispositivo non supporta 16 kHz, il backend registra alla frequenza nativa e ricampiona da solo.
- **Cartella Documenti su OneDrive**: il percorso di archiviazione si può cambiare dalle Impostazioni dell'app.
- **Eliminazione modelli non riuscita**: chiudi le altre istanze del backend. Su Windows un file del modello resta bloccato finché è caricato in memoria; l'app scarica i modelli prima di eliminarli.

---

## 🔒 Privacy & Sicurezza
Tutti i modelli, i file audio e i vettori risiedono **esclusivamente sul tuo disco locale** (di default in `~/Documents/SbobinAI`). Nessun dato audio, testo o credenziale lascia il tuo Mac.

---

## 📄 Licenza
Rilasciato sotto licenza MIT.
