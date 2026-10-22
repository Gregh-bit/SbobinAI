"""
live_session.py — SbobinAI
Gestisce la sessione live attiva, il Live Ring Buffer (RAM) per query contestuali,
e coordina l'AudioEngine e il Transcriber.
"""
from __future__ import annotations

import asyncio
import logging
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
            "suggestedAnswer": suggested_answer or "Domanda rilevata dal docente durante la lezione.",
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
        """Avvia la cattura audio."""
        self.audio_engine.start()

    def push_transcript(self, text: str) -> None:
        """Aggiunge un blocco definitivo di testo al ring buffer e alla trascrizione integrale."""
        if not text or not text.strip():
            return
        clean_text = text.strip()
        now = time.time()
        chunk = TranscriptChunk(text=clean_text, timestamp=now)
        self._buffer.append(chunk)
        self.full_transcript.append(clean_text)

        # Rimuovi chunk più vecchi di RING_BUFFER_SECONDS
        cutoff = now - RING_BUFFER_SECONDS
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def push_chunk(self, text: str) -> None:
        """Alias compatibilità per push_transcript."""
        self.push_transcript(text)

    def get_recent_context(self, minutes: int = RING_BUFFER_MINUTES) -> str:
        """Restituisce il testo degli ultimi `minutes` minuti come stringa unica."""
        cutoff = time.time() - (minutes * 60)
        # Snapshot sicuro per evitare RuntimeError: deque mutated during iteration
        snapshot = list(self._buffer)
        recent = [c.text for c in snapshot if c.timestamp >= cutoff]
        return " ".join(recent)

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

        return audio_path


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
    """Avvia (o riavvia) una sessione live per una materia."""
    # Se ce n'era una precedente non chiusa, chiudila per liberare risorse audio
    if subject_id in _active_sessions:
        end_session(subject_id)

    session = LiveSession(
        subject_id=subject_id,
        lecture_id=lecture_id,
        whisper_keywords=whisper_keywords,
        mock_transcriber=mock_transcriber,
        model_name=model_name,
        vad_sensitivity=vad_sensitivity,
    )
    session.start_audio()
    _active_sessions[subject_id] = session
    return session


def get_session(subject_id: int) -> Optional[LiveSession]:
    return _active_sessions.get(subject_id)


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
