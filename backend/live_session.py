"""
live_session.py — SbobinAI
Gestisce la sessione live attiva, il Live Ring Buffer (RAM) per query contestuali,
e coordina l'AudioEngine e il Transcriber.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

from audio_engine import AudioEngine
from transcriber import Transcriber

logger = logging.getLogger("sbobinai.session")

# ── Costanti ───────────────────────────────────────────────────────────────────
RING_BUFFER_MINUTES = 15  # ultimi N minuti tenuti in RAM
RING_BUFFER_SECONDS = RING_BUFFER_MINUTES * 60


@dataclass
class TranscriptChunk:
    text: str
    timestamp: float = field(default_factory=time.time)  # epoch seconds


class LiveSession:
    """
    Rappresenta una sessione di lezione live attiva.
    - Mantiene l'istanza dell'AudioEngine (cattura mic in RAM)
    - Mantiene l'istanza del Transcriber (in mock mode o Whisper)
    - Mantiene un ring-buffer circolare in RAM degli ultimi 15 minuti di testo
    - Conserva la trascrizione integrale per il salvataggio su DB
    """

    def __init__(
        self,
        subject_id: int,
        lecture_id: int,
        whisper_keywords: Optional[str] = None,
        mock_transcriber: bool = False,
        model_name: Optional[str] = None,
        vad_sensitivity: int = 2,
    ):
        self.subject_id = subject_id
        self.lecture_id = lecture_id
        self.whisper_keywords = whisper_keywords
        self.started_at = time.time()
        self.is_active: bool = True
        self.is_paused: bool = False
        # True quando la sessione è stata chiusa e persistita (evita doppio salvataggio)
        self.finalized: bool = False
        self._pause_started_at: float = 0.0
        self._total_paused_duration: float = 0.0

        self._buffer: Deque[TranscriptChunk] = deque()
        self.full_transcript: list[str] = []
        self.detected_questions: list[dict] = []

        # File temporaneo su disco per recupero progressivo in caso di crash/spegnimento
        try:
            from config import get_audio_dir
            audio_dir = get_audio_dir()
            os.makedirs(audio_dir, exist_ok=True)
            self.recovery_transcript_path = os.path.join(audio_dir, f"lecture_{lecture_id}.partial.txt")
        except Exception:
            self.recovery_transcript_path = None

        # Motore audio in RAM con VAD configurato
        self.audio_engine: AudioEngine = AudioEngine(
            lecture_id=lecture_id,
            vad_aggressiveness=vad_sensitivity,
        )
        # Transcriber (MLX Whisper o Mock Fallback)
        self.transcriber: Transcriber = Transcriber(
            model_name=model_name,
            mock_mode=mock_transcriber,
            whisper_keywords=whisper_keywords,
        )

        # Task asincrono per lo streaming della trascrizione
        self.transcription_task: Optional[asyncio.Task] = None
        self._client_disconnected_at: Optional[float] = None
        self._disconnect_grace_task: Optional[asyncio.Task] = None

    def add_question(self, question_text: str, confidence: float = 0.95, suggested_answer: str = "") -> dict:
        """Registra una domanda docente rilevata, evitando duplicati."""
        clean_new = question_text.lower().replace("?", "").replace(".", "").strip()
        for existing in self.detected_questions:
            clean_existing = existing["question"].lower().replace("?", "").replace(".", "").strip()
            if clean_new == clean_existing or (len(clean_new) > 15 and clean_new in clean_existing):
                return existing

        q_obj = {
            "id": f"q-{len(self.detected_questions) + 1}-{int(time.time())}",
            "question": question_text,
            "confidence": confidence,
            "suggestedAnswer": suggested_answer,
            "sourceLecture": "Questa lezione",
            "timestamp": int(time.time() * 1000),
        }
        self.detected_questions.append(q_obj)
        return q_obj

    def delete_question(self, question_id: str) -> bool:
        """Rimuove una domanda rilevata durante la sessione live tramite il suo id."""
        initial_len = len(self.detected_questions)
        self.detected_questions = [q for q in self.detected_questions if q.get("id") != question_id]
        return len(self.detected_questions) < initial_len

    def start_audio(self) -> None:
        """Avvia la cattura audio e sincronizza il timestamp d'inizio effettivo."""
        self.audio_engine.start()
        self.started_at = self.audio_engine.started_at

    def push_transcript(self, text: str) -> None:
        """Aggiunge un blocco definitivo di testo al ring buffer e alla trascrizione integrale."""
        if not text or not text.strip():
            return
        clean_text = text.strip()
        if self.full_transcript and clean_text.lower() == self.full_transcript[-1].lower():
            logger.debug(f"[LiveSession] Scartato chunk duplicato consecutivo: '{clean_text}'")
            return
        now = self.elapsed_seconds()
        chunk = TranscriptChunk(text=clean_text, timestamp=now)
        self._buffer.append(chunk)
        self.full_transcript.append(clean_text)

        # Scrittura progressiva su disco (recovery anti-crash)
        if self.recovery_transcript_path:
            try:
                with open(self.recovery_transcript_path, "a", encoding="utf-8") as rf:
                    rf.write(f"{clean_text}\n")
            except Exception:
                pass

        # Rimuovi chunk piu vecchi di RING_BUFFER_SECONDS calcolati sul tempo effettivo di lezione
        cutoff = now - RING_BUFFER_SECONDS
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def push_chunk(self, text: str) -> None:
        """Alias compatibilita per push_transcript."""
        self.push_transcript(text)

    def get_recent_context(self, minutes: int = RING_BUFFER_MINUTES) -> str:
        """Restituisce il testo degli ultimi `minutes` minuti di lezione effettiva."""
        cutoff = self.elapsed_seconds() - (minutes * 60)
        # Snapshot sicuro per evitare RuntimeError: deque mutated during iteration
        snapshot = list(self._buffer)
        recent = [c.text for c in snapshot if c.timestamp >= cutoff]
        return " ".join(recent)

    def get_timestamped_recent_context(
        self,
        minutes: int = RING_BUFFER_MINUTES,
        max_chars: int = 4500,
    ) -> str:
        """
        Restituisce la trascrizione live degli ultimi `minutes` minuti annotata con
        timestamp assoluto [MM:SS] e distanza temporale relativa (es. '2 min fa'),
        così l'LLM risponde con precisione anche a domande temporali ('cosa ho detto 2 minuti fa?').
        """
        now = self.elapsed_seconds()
        cutoff = now - (minutes * 60)
        snapshot = list(self._buffer)
        recent = [c for c in snapshot if c.timestamp >= cutoff and c.text.strip()]
        if not recent and self.full_transcript:
            # Fallback se il buffer circolare è vuoto ma esiste testo trascritto
            tail_text = " ".join(self.full_transcript[-20:]).strip()
            return tail_text[-max_chars:] if tail_text else ""
        if not recent:
            return ""

        def _fmt_mmss(sec: float) -> str:
            total = max(0, int(round(sec)))
            m, s = divmod(total, 60)
            return f"{m:02d}:{s:02d}"

        def _fmt_ago(delta_sec: float) -> str:
            delta = max(0, int(round(delta_sec)))
            if delta < 15:
                return "appena detto"
            if delta < 60:
                return f"{delta} sec fa"
            mins = int(round(delta / 60.0))
            return f"{mins} min fa"

        header = f"[Tempo attuale della lezione: {_fmt_mmss(now)}]"
        lines: list[str] = []
        used_chars = len(header) + 1
        # Accumula dal più recente al più vecchio rispettando max_chars, poi inverti in ordine cronologico
        for chunk in reversed(recent):
            delta = max(0.0, now - chunk.timestamp)
            line = f"[{_fmt_mmss(chunk.timestamp)} | {_fmt_ago(delta)}] {chunk.text.strip()}"
            if lines and used_chars + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used_chars += len(line) + 1

        lines.reverse()
        return header + "\n" + "\n".join(lines)

    def get_full_transcript(self) -> str:
        return " ".join(self.full_transcript)

    def elapsed_seconds(self) -> float:
        current_pause = (time.time() - self._pause_started_at) if self.is_paused else 0.0
        return max(0.0, time.time() - self.started_at - self._total_paused_duration - current_pause)

    def pause(self) -> Optional[bytes]:
        """Sospende la sessione live e restituisce l'ultimo audio residuo da consolidare con Large."""
        if not self.is_active or self.is_paused:
            return None
        self.is_paused = True
        self._pause_started_at = time.time()
        return self.audio_engine.pause()

    def resume(self) -> bool:
        """Riprende la sessione live e riattiva il microfono."""
        if not self.is_active or not self.is_paused:
            return False
        if self._pause_started_at > 0:
            self._total_paused_duration += time.time() - self._pause_started_at
            self._pause_started_at = 0.0
        self.is_paused = False
        return self.audio_engine.resume()

    def stop(self) -> Optional[str]:
        """
        Interrompe la sessione live.
        Arresta l'audio engine, esporta il WAV su disco e cancella il task di trascrizione.
        Restituisce il percorso del file audio generato.
        """
        self.is_active = False

        if self.transcription_task and not self.transcription_task.done():
            self.transcription_task.cancel()

        audio_path = None
        if self.audio_engine.is_running:
            audio_path = self.audio_engine.stop()

        if hasattr(self, "transcriber") and self.transcriber:
            try:
                self.transcriber.release_memory()
            except Exception as e:
                logger.debug(f"[LiveSession] Errore release memory: {e}")

        # Rimozione file recovery temporaneo al completamento pulito della sessione
        if getattr(self, "recovery_transcript_path", None) and os.path.exists(self.recovery_transcript_path):
            try:
                os.remove(self.recovery_transcript_path)
            except Exception:
                pass

        return audio_path

    def update_vad_sensitivity(self, level: int) -> None:
        """Aggiorna in tempo reale la sensibilità del VAD e la soglia RMS sulla registrazione attiva."""
        if hasattr(self, "audio_engine") and self.audio_engine:
            self.audio_engine.set_vad_sensitivity(level)


# ── Registry globale delle sessioni attive ──────────────────────────────────────
# chiave: subject_id → sessione corrente
_active_sessions: Dict[int, LiveSession] = {}


def start_session(
    subject_id: int,
    lecture_id: int,
    whisper_keywords: Optional[str] = None,
    mock_transcriber: bool = False,
    model_name: Optional[str] = None,
    vad_sensitivity: int = 2,
) -> LiveSession:
    """Avvia (o riavvia) una sessione live per una materia garantendo una sola registrazione attiva nell'app."""
    # Se c'erano sessioni precedenti aperte su qualsiasi materia, chiudile per liberare il microfono hardware
    for old_sid in list(_active_sessions.keys()):
        try:
            end_session(old_sid)
        except Exception:
            pass

    session = LiveSession(
        subject_id=subject_id,
        lecture_id=lecture_id,
        whisper_keywords=whisper_keywords,
        mock_transcriber=mock_transcriber,
        model_name=model_name,
        vad_sensitivity=vad_sensitivity,
    )
    session.start_audio()
    # Pre-riscaldamento asincrono immediato: carica Whisper in RAM prima dell'arrivo del parlato
    if hasattr(session, "transcriber") and session.transcriber:
        try:
            session.transcriber.preload_model_background()
        except Exception as e:
            logger.debug(f"[LiveSession] Preload background: {e}")

    _active_sessions[subject_id] = session
    return session


def get_session(subject_id: int) -> Optional[LiveSession]:
    return _active_sessions.get(subject_id)


def get_all_sessions() -> Dict[int, LiveSession]:
    """Restituisce un dizionario con tutte le sessioni live registrate."""
    return dict(_active_sessions)


def has_active_sessions() -> bool:
    """True se almeno una registrazione live è in corso (su qualsiasi materia)."""
    return any(s.is_active for s in _active_sessions.values())


def end_session(
    subject_id: int, session: Optional[LiveSession] = None
) -> Optional[tuple[LiveSession, Optional[str]]]:
    """
    Termina la sessione, salva il file audio ed elimina la sessione dal registry.
    Se `session` è indicata, termina solo se è ancora quella registrata per la materia
    (evita che la disconnessione di un socket vecchio chiuda la sessione di uno nuovo).
    """
    current = _active_sessions.get(subject_id)
    if current is None or (session is not None and current is not session):
        return None
    _active_sessions.pop(subject_id, None)
    audio_path = current.stop()
    return current, audio_path
