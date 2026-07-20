"""
src/pattern_filter.py
Module 1A: Lexical Filter (Plane 1)
---------------------------------------------------
Fast, deterministic, regex-based pre-inference check.
Upgraded to detect RAG delimiter smuggling and hidden steganography.
"""

import re
import base64

# ── Regex Blacklists (Direct Injection / Direct Jailbreak coverage) ──────────
OVERRIDE_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|preceding) instructions?",
    r"disregard (all |any |the )?(previous|prior|above|preceding) instructions?",
    r"forget (all |everything|your) (instructions?|rules|guidelines|training)",
    r"new instructions?:",
    r"system prompt",
    r"reveal (your |the )?(system prompt|instructions|prompt)",
    r"override (your |the )?(rules|guidelines|restrictions|filters)",
    r"bypass (your |the )?(rules|guidelines|restrictions|filters|safety)",
]

ROLEPLAY_TRIGGER_PATTERNS = [
    r"\byou are now\b",
    r"\bpretend (you are|to be)\b",
    r"\back as (if |a |an )?",
    r"\bDAN\b",                                   
    r"\bjailbreak(ed|ing)?\b",
    r"\bno (content policy|restrictions|filters|rules)\b",
    r"\bunfiltered (AI|version|mode)\b",
    r"\bstay in character\b",
    r"\bnever break character\b",
    r"\bfrom now on you (will|must)\b",
]

# 🟢 NEW: RAG Poisoning & Delimiter Smuggling
RAG_POISON_PATTERNS = [
    r"\[\s*SYSTEM NOTE\s*:",           # Explicitly catches Microsoft BIPIA
    r"\[\s*System Instructions?\s*:",  
    r"<!--.*?-->",                     # HTML comments hiding instructions
    r"[\u200B\u200C\u200D\uFEFF]{2,}", # Zero-width character steganography
]

# Category labels feed into the "Attack vector/type" field logged by BLOCK & LOG
CATEGORY_LABELS = {
    "override": OVERRIDE_PATTERNS,
    "roleplay": ROLEPLAY_TRIGGER_PATTERNS,
    "rag_poisoning": RAG_POISON_PATTERNS, # 🟢 Wired in
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE | re.DOTALL) for p in patterns]
    for cat, patterns in CATEGORY_LABELS.items()
}


# ── Base64 / Hex Deobfuscation ────────────────────────────────────────────────
def decode_base64_hex(text: str) -> str:
    b64_pattern = re.compile(r'[A-Za-z0-9+/]{20,}={0,2}')
    for match in b64_pattern.findall(text):
        try:
            decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
            if decoded.isprintable() and len(decoded) > 5:
                text = text.replace(match, decoded)
        except Exception:
            pass

    hex_pattern = re.compile(r'(?:0x)?[0-9a-fA-F]{8,}')
    for match in hex_pattern.findall(text):
        try:
            clean = match.replace("0x", "")
            decoded = bytes.fromhex(clean).decode("utf-8", errors="ignore")
            if decoded.isprintable() and len(decoded) > 3:
                text = text.replace(match, decoded)
        except Exception:
            pass

    return text


# ── Main entry point ──────────────────────────────────────────────────────────
def check_lexical(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {"score_1A": 0, "matched_category": None, "matched_pattern": None,
                "deobfuscated_text": text}

    deobfuscated = decode_base64_hex(text)

    for category, compiled_patterns in _COMPILED.items():
        for pattern in compiled_patterns:
            match = pattern.search(deobfuscated)
            if match:
                return {
                    "score_1A": 1,
                    "matched_category": category,
                    "matched_pattern": match.group(0),
                    "deobfuscated_text": deobfuscated,
                }

    return {"score_1A": 0, "matched_category": None, "matched_pattern": None,
            "deobfuscated_text": deobfuscated}