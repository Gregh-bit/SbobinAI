# SbobinAI 🎓

> **L'assistente universitario intelligente, 100% locale e privato.**  
> Trasforma registrazioni, slide e appunti in quaderni di studio interattivi, quiz d'esame e simulazioni orali con intelligenza artificiale eseguita interamente sul tuo dispositivo.

[![macOS](https://img.shields.io/badge/macOS-Apple%20Silicon%20(Metal)-black?logo=apple&style=flat-square)](https://github.com/Gregh-bit/SbobinAI/releases)
[![Windows](https://img.shields.io/badge/Windows-10%20%2F%2011%20x64-0078d4?logo=windows&style=flat-square)](https://github.com/Gregh-bit/SbobinAI/releases)
[![Downloads](https://img.shields.io/github/downloads/Gregh-bit/SbobinAI/total?color=2563eb&style=flat-square&logo=github&label=Downloads)](https://github.com/Gregh-bit/SbobinAI/releases)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20Locale-success?style=flat-square)](https://github.com/Gregh-bit/SbobinAI)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

---

## 📥 Scarica SbobinAI

Puoi scaricare l'applicazione precompilata scegliendo il canale che preferisci:

### 🌐 Dal Sito Ufficiale
👉 **[Accedi alla pagina ufficiale di SbobinAI](https://gregfolio.netlify.app/progetti/sbobinai)**  
*(Riconosce in automatico il tuo sistema operativo e avvia il download dell'ultima versione)*

### 📦 Download Diretto da GitHub
* 🍏 **[Scarica per macOS (.dmg)](https://github.com/Gregh-bit/SbobinAI/releases/latest/download/SbobinAI-macOS.dmg)** — Ottimizzato per Apple Silicon (M1/M2/M3/M4)
* 🪟 **[Scarica per Windows (.exe)](https://github.com/Gregh-bit/SbobinAI/releases/latest/download/SbobinAI-Windows.exe)** — Installer per Windows 10/11 x64

> [!TIP]
> **Nota per gli utenti macOS (Apple Silicon, macOS Sonoma e Sequoia):**  
> Trattandosi di un software open source distribuito gratuitamente al di fuori del Mac App Store, al primo avvio macOS Gatekeeper potrebbe mostrare l'avviso *"SbobinAI è danneggiato e non può essere aperto. Dovresti spostarlo nel Cestino."*  
> **Risoluzione immediata:** dopo aver trascinato l'app in **Applicazioni**, apri il **Terminale** e digita:
> ```bash
> xattr -cr /Applications/SbobinAI.app
> ```
> Premi Invio: l'attributo di quarantena web verrà rimosso e potrai aprire SbobinAI normalmente con un doppio clic.

Puoi visualizzare l'elenco di tutte le versioni, i changelog e i file di installazione nella [sezione GitHub Releases](https://github.com/Gregh-bit/SbobinAI/releases).

---

## ✨ Funzionalità

- 🎙️ **Registrazione Live & Trascrizione Streaming**: Trascrive la voce in tempo reale ad alta precisione con Whisper Large-v3-Turbo a 4-bit, con buffering in memoria a zero carico sul disco.
- 🔖 **Segnalibri Esame (`⌘B` / `⌘⇧B`) & Player Audio Sincronizzato**: Segna con una scorciatoia i punti cruciali indicati dal professore e riascoltali con un clic al secondo esatto (`click-to-play`).
- 📖 **Quaderno di Studio Intelligente**: Genera automaticamente capitoli ordinati, formule e schemi Markdown completi basati sulle lezioni. Esportabile con un clic.
- 📚 **Ingestione dei Materiali (RAG Ibrido)**: Carica slide, PDF e audio esterni; il motore indicizza i contenuti su LanceDB rendendoli consultabili all'istante dall'assistente.
- 🧠 **Quiz & Simulazione Esame Orale**:
  - **Quiz a Risposta Multipla**: 5 domande contestualizzate per lezione con punteggio e spiegazione.
  - **Interrogazione Vocale**: Rispondi a voce al microfono a una domanda aperta; l'IA valuta la tua esposizione in trentesimi, evidenziando punti di forza e lacune.
- 🔒 **Zero Cloud & Zero Swap**: Nessun abbonamento, nessuna chiave API, nessun dato lascia il tuo computer.

---

## 🛠️ Architettura & Stack Tecnologico

SbobinAI seleziona all'avvio il motore di intelligenza artificiale più efficiente per l'hardware in uso:

| Componente | macOS (Apple Silicon) | Windows / Altri (x64) |
|---|---|---|
| **Interfaccia Desktop** | Tauri 2.0 (Rust) + React 18, Vite | Tauri 2.0 (Rust) + React 18, Vite |
| **Server Backend** | FastAPI (Python 3.12) locale | FastAPI (Python 3.12) locale |
| **Speech-to-Text** | `mlx-whisper` (Metal GPU/Neural Engine) | `faster-whisper` (CTranslate2, CUDA/CPU) |
| **Modello LLM** | `mlx-lm` (Qwen 3.5 4B quantizzato 4-bit) | `llama-cpp-python` (Qwen 3.5 4B GGUF Q4) |
| **Vector DB & RAG** | LanceDB + Qwen3-Embedding (0.6B) | LanceDB + Qwen3-Embedding GGUF |
| **Database Dati** | SQLite con SQLAlchemy ORM | SQLite con SQLAlchemy ORM |

---

## 💻 Avvio in Sviluppo (Per Sviluppatori)

Se desideri clonare e compilare SbobinAI in autonomia:

### 1. Prerequisiti
- **Python 3.12**
- **Node.js 18+** e `npm`
- **Rust** (installato via `rustup`)

### 2. Clona il repository
```bash
git clone https://github.com/Gregh-bit/SbobinAI.git
cd SbobinAI
```

### 3. Avvia Backend & Frontend
```bash
# Terminale 1: Backend Python
cd backend
python3 -m venv venv
source venv/bin/activate       # Su Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# Terminale 2: Frontend Desktop (Tauri)
cd frontend
npm install
npm run tauri dev
```

---

## 🔒 Privacy & Riservatezza

Tutti i modelli, i file audio, i documenti caricati e le trascrizioni risiedono **esclusivamente sul tuo disco locale** (nella cartella configurabile di SbobinAI, di default in Documenti). Non vengono effettuate chiamate a server esterni di terze parti né raccolti dati telemetrici.

---

## 📄 Licenza

Distribuito sotto licenza [MIT](LICENSE). Libero per uso personale e studio.
