"""
transcript_cleaner.py — Pulizia non distruttiva delle trascrizioni Whisper per LLM e RAG.

Alleggerisce il testo orale (intercalari isolati, esitazioni vocali, pause, punteggiatura
spezzata e ripetizioni/balbettii tipici di Whisper) PRIMA dell'invio ai prompt LLM e al
chunking vettoriale RAG, riducendo i token consumati senza modificare `raw_transcript` su DB.
"""
from __future__ import annotations

import re

# ── Pattern precompilati ───────────────────────────────────────────────────────

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]+")
_NON_BREAKING_SPACE_RE = re.compile(r"[\u00a0\u202f]")

# Tag descrittivi non parlati generati talvolta da Whisper ([Musica], (applausi), ...)
_NON_SPEECH_TAG_RE = re.compile(
    r"[\[\(]\s*(?:musica|music|applausi|applause|risate|laughter|silenzio|silence|"
    r"pausa|pause|rumore|noise|tosse|sottofondo|inaudibile|sigh)\s*[\]\)]",
    re.IGNORECASE,
)

# Boilerplate noto di allucinazione Whisper su pause finali
_BOILERPLATE_RE = re.compile(
    r"(?:sottotitoli\s+(?:creati|a\s+cura)\s+(?:dalla\s+comunità\s+amara\.org|di\s+[^\n.!?]+)|"
    r"grazie\s+(?:a\s+tutti|mille|di\s+tutto|a\s+tutti\s+voi)|"
    r"grazie\s+(?:per|dell['’])\s*(?:la\s+visione|l['’]attenzione|l['’]ascolto|aver\s+guardato|averci\s+seguito)|"
    r"(?:iscriviti|iscrivetevi)\s+al\s+canale|arrivederci)[.!?]*",
    re.IGNORECASE,
)

# Esitazioni vocali pure (mai parole semantiche come "è", "e", ecc.)
_PURE_HESITATIONS_RE = re.compile(
    r"\b(?:e+h+m+|u+h+m+|a+h+m+|u+m+m+|e+m+m+|m+m+m+|e+h{2,}|a+h{2,}|o+h{2,}|u+h+)\b",
    re.IGNORECASE,
)
_ISOLATED_EH_RE = re.compile(r"(?:(?<=^)|(?<=[.!?;:,\n]))\s*\beh+\b(?=\s*[,.!?;:\n]|$)", re.IGNORECASE)

# Intercalari ad inizio frase seguiti da pausa/virgola (es. "Allora, ragazzi, diciamo, oggi...")
_OPENING_CRUTCH_RE = re.compile(
    r"(^|[.!?;:\n]\s*)(?:(?:allora|diciamo(?:\s+così)?|praticamente|cioè|ecco|vabbè|va\s+bene|"
    r"ok|okay|insomma|niente|ragazzi|sentite|guardate|vediamo|per\s+così\s+dire|"
    r"come\s+(?:posso\s+)?dire)\s*(?:,\s*)+)+",
    re.IGNORECASE,
)

# Intercalari parentetici tra due virgole: ", diciamo," -> " "
_PAREN_PAIR_CRUTCH_RE = re.compile(
    r",\s*(?:allora|diciamo(?:\s+così)?|cioè|praticamente|vabbè|va\s+bene|ok|okay|"
    r"ecco|insomma|per\s+così\s+dire|come\s+(?:posso\s+)?dire|se\s+vogliamo|"
    r"capite|sai|sapete)\s*,",
    re.IGNORECASE,
)
# Intercalare finale prima di punteggiatura forte: ", no." / ", diciamo." -> "."
_PAREN_END_CRUTCH_RE = re.compile(
    r",\s*(?:allora|diciamo(?:\s+così)?|praticamente|vabbè|va\s+bene|ok|okay|"
    r"ecco|insomma|per\s+così\s+dire|come\s+(?:posso\s+)?dire|se\s+vogliamo|"
    r"capite|sai|sapete)\s*(?=[;.!?])",
    re.IGNORECASE,
)

# Espressioni riempitive sicure ovunque (protegge "come posso dire che ...")
_SAFE_FILLER_PHRASES_RE = re.compile(
    r"\b(?:come\s+posso\s+dire(?!\s+che\b)|diciamo\s+così|per\s+così\s+dire|un\s+attimo\s+solo|mi\s+sentite)\b",
    re.IGNORECASE,
)

# Pause e punti di sospensione multipli
_ELLIPSIS_RE = re.compile(r"(?:\.{2,}|…+)")
_ORPHAN_PAUSE_DOTS_RE = re.compile(r"\s+\.\s+(?=[a-zà-ÿ])")

# Ripetizioni immediate di singole parole (2+ lettere anche con virgola, 1 lettera solo con spazi)
_REPEATED_WORD_RE = re.compile(r"\b([A-Za-zÀ-ÿ]{2,24})\b(?:[\s,]+\1\b)+", re.IGNORECASE)
_REPEATED_1CHAR_WS_RE = re.compile(r"\b([A-Za-zÀ-ÿ])\b(?:[ \t]+\1\b)+", re.IGNORECASE)
_INTENTIONAL_DOUBLES = frozenset({"piano", "mano", "via", "poco", "passo", "tempo", "giorno"})

# Ripetizioni immediate di brevi sequenze da 2 a 6 parole (balbettii/loop Whisper)
_REPEATED_PHRASE_RE = re.compile(
    r"\b([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’]*(?:\s+[A-Za-zÀ-ÿ0-9'’]+){1,5})\b(?:[\s,;]+\1\b)+",
    re.IGNORECASE,
)

# Punteggiatura e spaziatura finale (preserva virgola decimale es. "3,14")
_COMMA_BEFORE_END_RE = re.compile(r",\s*([.!?])")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"(?:[;!?]|(?<!\d),)(?=[A-Za-zÀ-ÿ0-9])|,(?=[A-Za-zÀ-ÿ])")
_SENTENCE_CAP_RE = re.compile(r"(^|[.!?]\s+|\n+)([a-zà-ÿ])")


def _dedup_word(match: re.Match[str]) -> str:
    word = match.group(1)
    if word.lower() in _INTENTIONAL_DOUBLES:
        return match.group(0)
    return word


def _dedup_consecutive_sentences(text: str) -> str:
    """Collassa frasi identiche ripetute consecutivamente (tipico loop Whisper su silenzi)."""
    parts = re.split(r"([.!?\n]+)", text)
    out: list[str] = []
    prev_norm = ""

    i = 0
    while i < len(parts):
        clause = parts[i]
        delim = parts[i + 1] if i + 1 < len(parts) else ""
        i += 2

        norm = re.sub(r"[\W_]+", " ", clause).strip().lower()
        is_meaningful_clause = len(norm.split()) >= 1 and len(norm) >= 3
        if is_meaningful_clause and norm == prev_norm:
            # Mantieni eventuale a capo se la frase duplicata terminava un paragrafo
            if "\n" in delim and out and "\n" not in out[-1]:
                out.append("\n")
            continue
        if clause or delim:
            out.append(clause + delim)
        prev_norm = norm if is_meaningful_clause else ""
    return "".join(out)


def clean_transcript_for_llm(text: str) -> str:
    """
    Restituisce una copia pulita e compatta di una trascrizione Whisper per LLM e RAG.
    Preserva la struttura di paragrafi, numeri e lessico tecnico senza toccare l'originale.
    """
    if not text or not isinstance(text, str):
        return ""

    s = _ZERO_WIDTH_RE.sub("", text)
    s = _NON_BREAKING_SPACE_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # 1. Rimuove tag non vocali e frasi boilerplate note di Whisper
    s = _NON_SPEECH_TAG_RE.sub(" ", s)
    s = _BOILERPLATE_RE.sub(" ", s)

    # 2. Normalizza pause (...) prima di rimuovere intercalari così "ehm... diciamo..." combacia
    s = re.sub(r"(?:\.{2,}|…+)\s*$", ".", s)
    s = _ELLIPSIS_RE.sub(", ", s)
    s = _ORPHAN_PAUSE_DOTS_RE.sub(" ", s)

    # 3. Rimuove esitazioni vocali isolate
    s = _PURE_HESITATIONS_RE.sub(" ", s)
    s = _ISOLATED_EH_RE.sub("", s)

    # 4. Rimuove intercalari multipli in apertura frase o parentetici (es. "Allora, diciamo, oggi...")
    for _ in range(3):
        prev = s
        s = _OPENING_CRUTCH_RE.sub(r"\1", s)
        s = _PAREN_PAIR_CRUTCH_RE.sub(" ", s)
        s = _PAREN_END_CRUTCH_RE.sub("", s)
        if s == prev:
            break
    s = _SAFE_FILLER_PHRASES_RE.sub(" ", s)

    # 5. Collassa balbettii e ripetizioni immediate (parole, brevi frasi, frasi consecutive)
    s = _REPEATED_WORD_RE.sub(_dedup_word, s)
    s = _REPEATED_1CHAR_WS_RE.sub(r"\1", s)
    s = _REPEATED_PHRASE_RE.sub(r"\1", s)
    s = _dedup_consecutive_sentences(s)
    s = _REPEATED_WORD_RE.sub(_dedup_word, s)
    s = _REPEATED_1CHAR_WS_RE.sub(r"\1", s)

    # 6. Normalizza spazi e punteggiatura senza intaccare decimali o identificatori con punto
    s = re.sub(r"[ \t]+", " ", s)
    s = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", s)
    s = _COMMA_BEFORE_END_RE.sub(r"\1", s)
    s = re.sub(r",{2,}", ",", s)
    s = re.sub(r"([.!?])\s*,", r"\1", s)
    s = re.sub(r"(?:\.\s*){2,}", ". ", s)
    s = _SPACE_AFTER_PUNCT_RE.sub(r"\g<0> ", s)

    # Rimuove punteggiatura orfana all'inizio di riga/testo e spazi ai bordi riga
    lines = [re.sub(r"^[,;.:\s]+", "", ln).strip() for ln in s.split("\n")]
    s = "\n".join(ln for ln in lines if ln)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if s.endswith(","):
        s = s[:-1].rstrip() + "."

    # Ripristina eventuale maiuscola iniziale dopo aver rimosso "Allora, ..." a inizio frase
    s = _SENTENCE_CAP_RE.sub(lambda m: m.group(1) + m.group(2).upper(), s)
    return s
