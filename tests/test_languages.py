from src.languages import detect_language, resolve_language


def test_detects_indian_scripts():
    assert detect_language("मुझे बुखार है") == "hi"
    assert detect_language("আমার জ্বর হয়েছে") == "bn"
    assert detect_language("எனக்கு காய்ச்சல்") == "ta"
    assert detect_language("నాకు జ్వరం ఉంది") == "te"
    assert detect_language("ನನಗೆ ಜ್ವರ ಇದೆ") == "kn"
    assert detect_language("എനിക്ക് പനി ഉണ്ട്") == "ml"
    assert detect_language("મને તાવ છે") == "gu"
    assert detect_language("ਮੈਨੂੰ ਬੁਖਾਰ ਹੈ") == "pa"


def test_latin_text_is_english():
    assert detect_language("What causes fever?") == "en"
    assert detect_language("mujhe bukhar hai") == "en"   # Hinglish: pick Hindi in the menu


def test_explicit_choice_wins():
    assert resolve_language("hi", "mujhe bukhar hai") == "hi"
    assert resolve_language("auto", "मुझे बुखार है") == "hi"
    assert resolve_language("xx", "hello") == "en"
