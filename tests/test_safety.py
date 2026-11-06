"""Tests for the pre-AI safety rules. Run with:  python -m pytest -q"""
import pytest

from src.safety import CRISIS, DOSAGE, EMERGENCY, SAFE, check_message

CRISIS_CASES = [
    "I want to kill myself",
    "i dont want to live anymore",
    "I've been thinking about suicide",
    "my friend says she is suicidal",
    "I keep cutting myself",
    "Sometimes I feel like ending it all, I want to end it all",
    "I wish I was dead",
    "life is not worth living",
]

EMERGENCY_CASES = [
    ("I have chest pain", "heart"),
    ("My dad is having chest pain and sweating, what should I do?", "heart"),
    ("I think I'm having a heart attack", "heart"),
    ("my mother's face is drooping and her speech is slurred", "stroke"),
    ("My grandfather suddenly can't move his arm", "stroke"),
    ("I can't breathe", "breathing"),
    ("my baby is choking", "breathing"),
    ("I took too many sleeping pills", "overdose"),
    ("my son swallowed bleach", "overdose"),
    ("my friend overdosed", "overdose"),
    ("my husband collapsed and is unresponsive", "unconscious"),
    ("my brother is having a seizure right now", "seizure"),
    ("I'm bleeding heavily from a cut", "bleeding"),
    ("her throat is swelling after a bee sting", "allergy"),
]

# General questions: answered normally, but with an "if this is happening now" notice
NOTICE_CASES = [
    "What are the signs of a heart attack?",
    "What are the symptoms of a stroke?",
    "Is chest pain a symptom of anxiety?",
    "What happens if you overdose on paracetamol?",
    "What are the warning signs of suicide in teenagers?",
    "Can I get a stroke from high blood pressure?",
    "I had a stroke last year, how do I prevent another?",
    "I have a question about heart attacks",
]

DOSAGE_CASES = [
    "How much paracetamol should I take?",
    "how many ibuprofen can i take in a day",
    "What dose of amoxicillin for my 5 year old?",
    "what is the right dosage for me",
    "Is 500 mg of metformin ok for me?",
    "Can I double my dose of insulin?",
    "Should I stop taking my blood pressure tablets?",
    "Which medicine should I take for a headache?",
    "What should I take for a cold?",
    "Can you prescribe me some antibiotics?",
    "Suggest a medicine for fever",
]

NORMAL_CASES = [
    "What causes diabetes?",
    "What are the side effects of metformin?",
    "How does insulin work?",
    "What is asthma?",
    "Which drugs interact with warfarin?",
    "what are the symptoms of anemia",
    "Tell me about the treatment of bronchitis",
    "What is the mechanism of action of digoxin?",
]


@pytest.mark.parametrize("message", CRISIS_CASES)
def test_crisis(message):
    assert check_message(message).kind == CRISIS


@pytest.mark.parametrize("message,category", EMERGENCY_CASES)
def test_emergency(message, category):
    result = check_message(message)
    assert result.kind == EMERGENCY
    assert result.category == category


@pytest.mark.parametrize("message", NOTICE_CASES)
def test_general_question_gets_notice_not_block(message):
    result = check_message(message)
    assert result.kind == SAFE
    assert result.notice


@pytest.mark.parametrize("message", DOSAGE_CASES)
def test_dosage_refused(message):
    assert check_message(message).kind == DOSAGE


@pytest.mark.parametrize("message", NORMAL_CASES)
def test_normal_questions_pass(message):
    result = check_message(message)
    assert result.kind == SAFE
    assert result.notice is None
