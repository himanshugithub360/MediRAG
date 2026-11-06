"""Safety checks that run *before* the AI sees a message.

Three outcomes besides "safe":

- CRISIS: suicidal thoughts or self-harm. The AI is skipped and crisis helplines are shown.
- EMERGENCY: someone appears to be having a medical emergency right now (chest pain, stroke
  signs, can't breathe, overdose, ...). The AI is skipped and emergency numbers are shown.
- DOSAGE: a request for a personal dose or for which medicine to take. The AI is skipped and
  the user is pointed to a doctor or pharmacist.

General questions about the same topics ("what are the signs of a stroke?") are still
answered, but get a short "if this is happening now, call ..." notice attached.

These are keyword rules, so they are fast, free and predictable, but not perfect: they
err on the side of showing emergency information when a message is ambiguous.
"""
import re
from dataclasses import dataclass
from typing import Optional

from src.config import AMBULANCE_NUMBER, EMERGENCY_NUMBER, MENTAL_HEALTH_HELPLINE

SAFE, CRISIS, EMERGENCY, DOSAGE = "safe", "crisis", "emergency", "dosage"


@dataclass
class SafetyResult:
    kind: str                     # SAFE, CRISIS, EMERGENCY or DOSAGE
    category: str = ""            # e.g. "heart", "stroke"
    notice: Optional[str] = None  # extra note to show alongside a normal answer


def _rx(*patterns):
    return re.compile("|".join(f"(?:{p})" for p in patterns))


_PEOPLE = (r"friend|son|daughter|mother|mom|mum|father|dad|wife|husband|brother|sister|baby|child|"
           r"kid|grandma|grandpa|grandmother|grandfather|grandson|granddaughter|uncle|aunt|"
           r"neighbou?r|colleague|roommate|patient|boyfriend|girlfriend|partner")

# Someone specific is involved (the user, a relative, a friend, ...)
PERSONAL = _rx(
    r"\b(?:i|i'm|im|i've|ive|i'd|me|my|mine|myself|we|we're|our|us)\b",
    r"\b(?:he|he's|she|she's|him|her|his|they|they're|them|their|someone|somebody)\b",
    rf"\b(?:{_PEOPLE})\b",
)

# Someone is experiencing it: "my dad has ...", "I'm bleeding ...", "her throat is swelling"
PERSONAL_PRESENT = _rx(
    r"\b(?:i|i'm|im|i am|i've|ive|he|he's|she|she's|they|they're|we|we're|someone|somebody|"
    rf"(?:my|our|a) (?:\w+ ){{0,2}}?(?:{_PEOPLE})(?:'s)?)"
    r"\s+(?:\w+\s+){0,3}?(?:have|has|having|got|am|is|are|feel|feels|feeling|keep|keeps|"
    r"started|can't|cant|cannot|won't|suddenly|seems?|looks?|showing)\b",
    # present-continuous: "I'm bleeding", "she is choking"
    r"\b(?:i'm|im|i am|he's|she's|he is|she is|they're|they are|we're|we are)\b",
    # a body part or symptom of a specific person: "my face is", "her speech is", "his lips are"
    r"\b(?:my|his|her|their|our) (?:\w+ ){1,2}?(?:is|are|has|have|keeps|won't|feels|looks)\b",
)

# Strong signs that it is happening right now
URGENT = _rx(r"\b(?:right now|happening now|currently|suddenly|emergency|urgent|help me|please help|"
             r"what do i do|what should i do|getting worse)\b")

# The message is about the past, prevention, risk or study, not an emergency in progress
NOT_ACUTE = _rx(
    r"\b(?:last|previous|past) (?:year|month|week|time)\b",
    r"\b(?:years?|months?|weeks?) ago\b",
    r"\bhistory of\b",
    r"\b(?:had|after) (?:a|an|his|her|my|their) (?:\w+ )?(?:stroke|heart attack|seizure|"
    r"cardiac arrest|overdose|anaphyla\w*)",
    r"\brecover(?:y|ing|ed)?\b", r"\bprevent(?:ion|ing)?\b", r"\brisks?\b",
    r"\bsurviv(?:ed|or)\b",
    r"\b(?:a question|curious|wondering|just asking|in general|generally|exam|assignment|"
    r"homework|essay|studying|for a class)\b",
)

CRISIS_PATTERNS = _rx(
    r"\bkill (?:my ?self|myself)\b", r"\bend (?:my|my own) life\b", r"\bend it all\b",
    r"\btake my (?:own )?life\b", r"\b(?:want|wanna|going) to die\b", r"\bwish i (?:was|were) dead\b",
    r"\b(?:don't|dont|do not) want to (?:live|be alive|wake up)\b", r"\bno reason to live\b",
    r"\bbetter off dead\b", r"\b(?:hurt|harm|cut|cutting|burn) (?:my ?self|myself)\b",
    r"\bself[- ]?harm", r"\bnot worth living\b",
)
SUICIDE_WORD = re.compile(r"\bsuicid")
OVERDOSE_TOPIC = re.compile(r"\boverdos|\bpoison")

EMERGENCY_PATTERNS = {
    "heart": _rx(r"\bchest (?:pain|pains|tightness|pressure|hurts?|is hurting)\b",
                 r"\bpain in (?:my|his|her|the) chest\b", r"\bheart attacks?\b", r"\bcardiac arrest\b"),
    "stroke": _rx(r"\bstrokes?\b", r"\bface (?:is )?droop", r"\bdrooping face\b", r"\bslurred speech\b",
                  r"\bspeech (?:is )?slurr",
                  r"\bslurring\b", r"\bsudden (?:numbness|weakness|confusion|severe headache)\b",
                  r"\bone side of (?:my|his|her|the|their) (?:body|face)\b",
                  r"\bcan'?t (?:move|feel) (?:my|his|her|their) (?:arm|leg|face)\b"),
    "breathing": _rx(r"\b(?:can't|cant|cannot|unable to|struggling to|hard to) breathe?\b",
                     r"\bchoking\b", r"\blips (?:are |turning |look )?(?:blue|purple)\b",
                     r"\bgasping\b"),
    # An overdose or poisoning that has happened (the topic alone is matched by OVERDOSE_TOPIC)
    "overdose": _rx(r"\boverdosed\b", r"\b(?:has|have|had|just) overdos",
                    r"\btook (?:too many|too much|a whole|an entire|the whole|all (?:my|the|of|his|her))",
                    r"\b(?:swallowed|drank|drunk|ate|took|consumed|has taken|have taken) "
                    r"(?:some |the |a |an )?(?:\w+ )?(?:poison|bleach|pesticide|insecticide|"
                    r"kerosene|acid|phenyl|pills|tablets)\b",
                    r"\bpoisoned\b"),
    "unconscious": _rx(r"\bunconscious\b", r"\bnot breathing\b", r"\bunresponsive\b",
                       r"\bwon'?t wake up\b", r"\bcollapsed\b", r"\bpassed out\b", r"\bno pulse\b"),
    "seizure": _rx(r"\bseizures?\b", r"\bconvuls", r"\bhaving (?:a )?fits?\b"),
    "bleeding": _rx(r"\bbleeding (?:heavily|a lot|badly|won'?t stop|non ?stop)\b",
                    r"\b(?:severe|heavy|uncontrolled) bleeding\b", r"\b(?:lots?|a lot) of blood\b",
                    r"\bvomiting blood\b", r"\bcoughing (?:up )?blood\b"),
    "allergy": _rx(r"\bthroat (?:is )?(?:swelling|closing|tight)\b",
                   r"\b(?:tongue|lips|face) (?:is )?swelling\b", r"\banaphyla"),
}

_MEDICINE = r"(?:medicine|medication|drug|tablet|pill|antibiotic|syrup|cream|ointment|painkiller)s?"

DOSAGE_REQUEST = _rx(
    r"\bhow (?:much|many)\b.*\b(?:should|can|do|could|may|must) (?:i|he|she|we|they|my)\b",
    r"\bhow (?:much|many)\b.*\b(?:to|for) (?:take|give|me|my|him|her|us|them)\b",
    r"\b(?:dose|dosage|doses)\b.*\b(?:for (?:me|my|him|her|a \d+|an? (?:\d+|child|kid|baby|toddler|infant))|"
    r"should i|can i|do i|should he|should she|my (?:son|daughter|child|baby|kid))\b",
    r"\b(?:what|which|right|correct|safe|maximum|max)\b.*\b(?:dose|dosage)\b.*\b(?:should|can) (?:i|he|she|we)\b",
    r"\b\d+\s*(?:mg|ml|milligrams?|tablets?|pills?|capsules?)\b.*\b(?:should|can|is it (?:ok|okay|safe)"
    r"(?: for)?) (?:i|he|she|my|me)\b",
    r"\b\d+\s*(?:mg|ml|milligrams?|tablets?|pills?|capsules?)\b.*\bfor (?:me|my|him|her)\b",
    r"\b(?:increase|decrease|double|halve|change|skip|stop) (?:my|his|her) (?:dose|dosage|"
    r"medicine|medication|pills?|tablets?)\b",
    r"\b(?:should|can) (?:i|he|she) (?:stop|quit) taking\b",
)
PRESCRIPTION_REQUEST = _rx(
    r"\bprescribe (?:me|for me|something|my|him|her|a|an|some)\b",
    r"\b(?:write|give|get) (?:me |my |him |her )?(?:a )?prescription\b",
    rf"\b(?:which|what) {_MEDICINE} (?:should|can|do|could|must) (?:i|he|she|we|my)\b",
    r"\bwhat (?:should|can) (?:i|he|she|we) take\b",
    rf"\b(?:suggest|recommend|tell me) (?:a |an |some |any |the best )?{_MEDICINE}\b",
    rf"\bbest {_MEDICINE} for (?:me|my)\b",
)


def normalize(text):
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def check_message(message):
    """Classify a user message. Order matters: crisis > emergency > dosage."""
    text = normalize(message)
    personal = bool(PERSONAL.search(text))

    if CRISIS_PATTERNS.search(text) or (SUICIDE_WORD.search(text) and personal):
        return SafetyResult(CRISIS, "mental_health")
    if SUICIDE_WORD.search(text):
        return SafetyResult(SAFE, "mental_health", notice=crisis_notice())

    experiencing = bool(PERSONAL_PRESENT.search(text) or URGENT.search(text))
    for category, pattern in EMERGENCY_PATTERNS.items():
        if pattern.search(text):
            if category == "overdose":
                # "my son swallowed bleach" describes an event, so any named person makes it urgent
                acute = personal or experiencing
            else:
                acute = experiencing and not NOT_ACUTE.search(text)
            if acute:
                return SafetyResult(EMERGENCY, category)
            return SafetyResult(SAFE, category, notice=emergency_notice())
    if OVERDOSE_TOPIC.search(text):
        return SafetyResult(SAFE, "overdose", notice=emergency_notice())

    if DOSAGE_REQUEST.search(text) or PRESCRIPTION_REQUEST.search(text):
        return SafetyResult(DOSAGE)

    return SafetyResult(SAFE)


# ── Messages ─────────────────────────────────────────────────────────────────
EMERGENCY_ADVICE = {
    "heart": "Chest pain can be a sign of a heart attack. Stop any activity, sit down, stay calm and "
             "don't drive yourself to hospital. Follow the emergency operator's instructions.",
    "stroke": "These can be signs of a stroke. Note the time the symptoms started; treatment works "
              "best in the first few hours. Don't give food, drink or medicine.",
    "breathing": "Severe difficulty breathing needs immediate help. Sit upright and loosen tight clothing. "
                 "If they use a prescribed inhaler or adrenaline (epinephrine) pen, use it now.",
    "overdose": "Don't wait for symptoms to appear. Keep the medicine or chemical container to show the "
                "doctors, and don't try to make the person vomit unless a professional tells you to.",
    "unconscious": "If the person isn't breathing normally, start CPR (hard, fast chest compressions) "
                   "if you know how, and follow the emergency operator's instructions.",
    "seizure": "Move hard or sharp objects away, cushion their head, and turn them onto their side when "
               "the jerking stops. Don't put anything in their mouth. Get help if it lasts over "
               "5 minutes or they don't wake up.",
    "bleeding": "Press firmly on the wound with a clean cloth and keep pressing. Raise the injured part "
                "if possible.",
    "allergy": "Throat or face swelling can be a severe allergic reaction (anaphylaxis). If they have an "
               "adrenaline (epinephrine) auto-injector, use it now.",
}


def emergency_response(category):
    advice = EMERGENCY_ADVICE.get(category, "")
    return (
        "🚨 **This could be a medical emergency. Please get help right now.**\n\n"
        f"📞 Call **{EMERGENCY_NUMBER}** (emergency) or **{AMBULANCE_NUMBER}** (ambulance), "
        "or go to the nearest hospital emergency department.\n\n"
        f"{advice}\n\n"
        "I'm an AI and can't assess emergencies. Please don't wait for an online answer."
    )


def crisis_response():
    return (
        "💙 **I'm really sorry you're going through this. You don't have to face it alone.**\n\n"
        f"📞 Please talk to someone now: call **Tele-MANAS {MENTAL_HEALTH_HELPLINE}** "
        "(free, 24×7, confidential mental-health support), or "
        f"**{EMERGENCY_NUMBER}** if you or someone else is in immediate danger.\n\n"
        "If you can, reach out to someone you trust and stay with them, and keep away from anything "
        "you could use to hurt yourself.\n\n"
        "I'm an AI, so I can't give the support you deserve right now, but trained people at these "
        "numbers can, any time of day or night."
    )


def dosage_response():
    return (
        "💊 **I can't recommend medicines or doses for a specific person.**\n\n"
        "The right medicine and dose depend on age, weight, kidney and liver function, pregnancy, "
        "allergies and any other medicines being taken. Getting it wrong can be dangerous.\n\n"
        "Please ask a **doctor or pharmacist**. They can check these details safely, and always "
        "follow the dose on the prescription or package label.\n\n"
        "I *can* explain how a medicine works, what it's generally used for, its common side "
        "effects and interactions. Feel free to ask about that."
    )


def emergency_notice():
    return (f"If you or someone near you has these symptoms right now, call {EMERGENCY_NUMBER} "
            f"or {AMBULANCE_NUMBER} immediately.")


def crisis_notice():
    return (f"If you or someone you know is struggling, call Tele-MANAS {MENTAL_HEALTH_HELPLINE} "
            f"(free, 24×7) or {EMERGENCY_NUMBER} in an emergency.")


def low_confidence_response():
    return (
        "I couldn't find reliable information about that in my medical reference books, so I'd "
        "rather not guess. 🙏\n\n"
        "Try rephrasing your question with the specific condition, symptom or medicine name. For "
        "anything about your own health, please consult a doctor."
    )
