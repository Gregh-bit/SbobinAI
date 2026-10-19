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

## 🔒 Privacy & Sicurezza
Tutti i modelli, i file audio e i vettori risiedono **esclusivamente sul tuo disco locale** (di default in `~/Documents/SbobinAI`). Nessun dato audio, testo o credenziale lascia il tuo Mac.

---

## 📄 Licenza
Rilasciato sotto licenza MIT.
