"""Languages the chatbot can answer in.

Questions in another language are translated to English for searching (the reference
books are in English), and the answer is written back in the user's language.
"""
import re

# code: (English name, native name, browser speech locale)
LANGUAGES = {
    "en": ("English", "English", "en-IN"),
    "hi": ("Hindi", "हिन्दी", "hi-IN"),
    "bn": ("Bengali", "বাংলা", "bn-IN"),
    "mr": ("Marathi", "मराठी", "mr-IN"),
    "ta": ("Tamil", "தமிழ்", "ta-IN"),
    "te": ("Telugu", "తెలుగు", "te-IN"),
    "gu": ("Gujarati", "ગુજરાતી", "gu-IN"),
    "kn": ("Kannada", "ಕನ್ನಡ", "kn-IN"),
    "ml": ("Malayalam", "മലയാളം", "ml-IN"),
    "pa": ("Punjabi", "ਪੰਜਾਬੀ", "pa-IN"),
}

# Unicode blocks of Indian scripts -> language. Devanagari is shared by Hindi and Marathi;
# auto-detection assumes Hindi, so Marathi users should pick Marathi explicitly.
_SCRIPTS = [
    (re.compile(r"[ऀ-ॿ]"), "hi"),
    (re.compile(r"[ঀ-৿]"), "bn"),
    (re.compile(r"[਀-੿]"), "pa"),
    (re.compile(r"[઀-૿]"), "gu"),
    (re.compile(r"[஀-௿]"), "ta"),
    (re.compile(r"[ఀ-౿]"), "te"),
    (re.compile(r"[ಀ-೿]"), "kn"),
    (re.compile(r"[ഀ-ൿ]"), "ml"),
]


def detect_language(text):
    """Guess the language from the script used. Latin script (incl. Hinglish) -> English."""
    counts = [(len(pattern.findall(text)), code) for pattern, code in _SCRIPTS]
    best_count, best_code = max(counts)
    return best_code if best_count >= 2 else "en"


def resolve_language(requested, text):
    """The language to answer in: the user's explicit choice, or auto-detected."""
    if requested in LANGUAGES:
        return requested
    return detect_language(text)


def language_name(code):
    return LANGUAGES.get(code, LANGUAGES["en"])[0]


def public_list():
    return [{"code": c, "name": n, "native": nat, "speech": sp} for c, (n, nat, sp) in LANGUAGES.items()]
