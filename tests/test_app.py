"""API tests for the Flask app, using the fake services from conftest.py."""
import io
import json

import app as app_module
from tests.conftest import ask


# ── Greetings ────────────────────────────────────────────────────────────────
def test_pure_greeting_is_answered_without_ai(client, fake):
    status, data = ask(client, "Hi!")
    assert status == 200
    assert data["response"] == app_module.QUICK_RESPONSES["hi"]
    assert fake.qa_chain.calls == [] and fake.retriever.queries == []


def test_greeting_aliases():
    assert app_module.quick_reply("hello doctor") == app_module.QUICK_RESPONSES["hello"]
    assert app_module.quick_reply("Thanks a lot!!") == app_module.QUICK_RESPONSES["thanks"]
    assert app_module.quick_reply("namaste") == app_module.QUICK_RESPONSES["hello"]


def test_greeting_followed_by_question_gets_a_real_answer(client, fake):
    status, data = ask(client, "hey, what causes diabetes?")
    assert status == 200
    assert "Diabetes" in data["response"]
    assert len(fake.qa_chain.calls) == 1
    assert app_module.quick_reply("hey, what causes diabetes?") is None


# ── Normal answers ───────────────────────────────────────────────────────────
def test_answer_has_sources_and_is_saved(client, fake):
    status, data = ask(client, "What is diabetes?")
    assert status == 200 and data["kind"] is None
    assert data["sources"][0]["title"] == "MedlinePlus: Diabetes"
    assert data["sources"][0]["url"].startswith("https://medlineplus.gov")

    convs = client.get("/api/conversations").get_json()
    assert len(convs) == 1 and convs[0]["title"] == "What is diabetes?"
    messages = client.get(f"/api/conversations/{data['conversation_id']}").get_json()["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]


def test_follow_up_is_rewritten_before_searching(client, fake):
    _, first = ask(client, "What is diabetes?")
    fake.contextualize_chain.reply = "TRANSLATION: what are its symptoms?\nQUESTION: What are the symptoms of diabetes?"
    ask(client, "what are its symptoms?", conversation_id=first["conversation_id"])
    assert fake.retriever.queries[-1][0] == "What are the symptoms of diabetes?"
    history = fake.qa_chain.calls[-1]["chat_history"]
    assert len(history) == 2          # previous question + answer were sent as context


def test_source_filter_is_passed_to_retriever(client, fake):
    ask(client, "What is malaria?", collections=["who", "not-a-collection"])
    assert fake.retriever.queries[-1][1] == ["who"]


def test_hindi_question_is_translated_for_search_and_answered_in_hindi(client, fake):
    status, data = ask(client, "मधुमेह के लक्षण क्या हैं?")
    assert status == 200 and data["language"] == "hi"
    assert fake.retriever.queries[-1][0] == "What are the symptoms of diabetes?"
    assert fake.qa_chain.calls[-1]["language"] == "Hindi"


# ── Safety paths ─────────────────────────────────────────────────────────────
def test_emergency_skips_the_ai(client, fake):
    status, data = ask(client, "my dad is having chest pain")
    assert status == 200 and data["kind"] == "emergency"
    assert "112" in data["response"]
    assert fake.qa_chain.calls == [] and fake.retriever.queries == []


def test_hindi_emergency_is_caught_without_ai(client, fake):
    _, data = ask(client, "मेरे पिता को सीने में दर्द हो रहा है")
    assert data["kind"] == "emergency"
    assert fake.contextualize_chain.calls == []


def test_dosage_request_is_refused(client, fake):
    _, data = ask(client, "How much paracetamol should I take?")
    assert data["kind"] == "dosage"
    assert fake.qa_chain.calls == []


def test_low_relevance_declines_instead_of_guessing(client, fake):
    fake.retriever.score_value = -8.0
    _, data = ask(client, "What is the capital of France?")
    assert data["kind"] == "low_confidence"
    assert fake.qa_chain.calls == []


# ── Error handling ───────────────────────────────────────────────────────────
def test_empty_message():
    with app_module.app.test_client() as c:
        assert c.post("/chat", json={"message": "   "}).get_json()["response"] == "Please enter a message."
        assert c.post("/chat", data="not json").get_json()["response"] == "Please enter a message."


def test_llm_failure_returns_friendly_error_and_is_left_out_of_context(client, fake):
    fake.qa_chain.errors = [RuntimeError("Mistral is down")]
    _, data = ask(client, "What is asthma?")
    assert data["kind"] == "error"
    assert data["response"] == app_module.ERROR_ANSWER

    ask(client, "What is asthma?", conversation_id=data["conversation_id"])
    roles = [type(m).__name__ for m in fake.qa_chain.calls[-1]["chat_history"]]
    assert roles == ["HumanMessage"]        # the failed answer isn't sent back to the LLM


def test_mistral_rate_limit_is_retried(client, fake):
    fake.qa_chain.errors = [RuntimeError("Error response 429 while fetching")]
    _, data = ask(client, "What is asthma?")
    assert data["kind"] is None and "Diabetes" in data["response"]
    assert len(fake.qa_chain.calls) == 2


def test_rate_limit_returns_429_but_emergencies_get_through(client, fake):
    app_module.limiter.enabled = True
    app_module.limiter.reset()
    codes = [client.post("/chat", json={"message": "hi"}).status_code for _ in range(11)]
    assert codes[:10] == [200] * 10 and codes[10] == 429
    blocked = client.post("/chat", json={"message": "hi"}).get_json()
    assert blocked["kind"] == "rate_limited"
    assert client.post("/chat", json={"message": "I can't breathe"}).status_code == 200


def test_unknown_conversation_starts_a_new_one(client, fake):
    _, data = ask(client, "hi", conversation_id="does-not-exist")
    assert data["conversation_id"] != "does-not-exist"


# ── Streaming ────────────────────────────────────────────────────────────────
def parse_sse(body):
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        events.append((fields["event"], json.loads(fields["data"])))
    return events


def test_streaming_endpoint_sends_conversation_tokens_and_done(client, fake):
    response = client.post("/api/chat/stream", json={"message": "What is diabetes?"})
    assert response.mimetype == "text/event-stream"
    events = parse_sse(response.get_data(as_text=True))
    names = [e for e, _ in events]
    assert names[0] == "conversation" and names[-1] == "done"
    assert names.count("token") > 3
    text = "".join(d["text"] for e, d in events if e == "token").strip()
    assert text == events[-1][1]["content"]


# ── Symptom checker ──────────────────────────────────────────────────────────
def test_symptom_check_uses_the_symptom_prompt(client, fake):
    response = client.post("/api/symptom-check/stream", json={
        "symptoms": "runny nose and sneezing", "age": "30", "duration": "1–3 days", "red_flags": []})
    done = parse_sse(response.get_data(as_text=True))[-1][1]
    assert done["kind"] == "symptom_check" and "Possible causes" in done["content"]
    assert fake.qa_chain.calls == [] and len(fake.symptom_chain.calls) == 1


def test_symptom_check_red_flag_goes_straight_to_emergency(client, fake):
    response = client.post("/api/symptom-check/stream", json={
        "symptoms": "headache", "red_flags": ["stiff_neck_fever"]})
    done = parse_sse(response.get_data(as_text=True))[-1][1]
    assert done["kind"] == "emergency" and "meningitis" in done["content"]
    assert fake.symptom_chain.calls == []


def test_symptom_check_requires_symptoms(client):
    assert client.post("/api/symptom-check/stream", json={"symptoms": ""}).status_code == 400


# ── Uploads ──────────────────────────────────────────────────────────────────
def make_pdf(lines):
    content = "BT /F1 11 Tf 50 780 Td 15 TL " + " ".join(f"({l}) Tj T*" for l in lines) + " ET"
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            "/Resources << /Font << /F1 5 0 R >> >> >>",
            f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>"]
    out, offsets = "%PDF-1.4\n", []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{obj}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    return out.encode("latin-1")


REPORT = ["SAMPLE LAB REPORT", "Haemoglobin 11.2 g/dL (13.0 - 17.0) L",
          "Fasting glucose 132 mg/dL (70 - 99) H"] + [f"Comment line {i} for the sample report" for i in range(5)]


def test_upload_pdf_then_ask_about_it(client, fake):
    response = client.post("/api/uploads", data={"file": (io.BytesIO(make_pdf(REPORT)), "report.pdf")},
                           content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    data = response.get_json()
    assert data["attachment"]["kind"] == "pdf" and data["attachment"]["pages"] == 1

    _, answer = ask(client, "Which values are abnormal?", conversation_id=data["conversation_id"])
    assert "Haemoglobin 11.2" in fake.qa_chain.calls[-1]["uploads"]
    assert answer["sources"][0]["title"] == "📎 report.pdf"

    file = client.get(f"/api/attachments/{data['attachment']['attachment_id']}/file")
    assert file.status_code == 200 and file.data.startswith(b"%PDF")


def test_upload_image_is_read_by_the_vision_model(client, fake):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, format="PNG")
    response = client.post("/api/uploads", data={"file": (io.BytesIO(buf.getvalue()), "photo.png")},
                           content_type="multipart/form-data")
    data = response.get_json()
    assert response.status_code == 200 and data["attachment"]["kind"] == "image"
    assert "Haemoglobin" in data["reply"]["content"]
    assert data["reply"]["notice"]                         # image disclaimer
    assert len(fake.vision_llm.calls) == 1


def test_upload_rejects_wrong_types_and_fake_files(client):
    bad = client.post("/api/uploads", data={"file": (io.BytesIO(b"hello"), "notes.txt")},
                      content_type="multipart/form-data")
    assert bad.status_code == 400
    fake_png = client.post("/api/uploads", data={"file": (io.BytesIO(b"not an image"), "x.png")},
                           content_type="multipart/form-data")
    assert fake_png.status_code == 400 and "image" in fake_png.get_json()["error"]


# ── Privacy, feedback, health ────────────────────────────────────────────────
def test_other_browsers_cannot_see_a_conversation(client, fake):
    _, data = ask(client, "What is diabetes?")
    with app_module.app.test_client() as stranger:
        assert stranger.get(f"/api/conversations/{data['conversation_id']}").status_code == 404
        assert stranger.delete(f"/api/conversations/{data['conversation_id']}").status_code == 404
        assert stranger.post(f"/api/messages/{data['message_id']}/feedback", json={"value": -1}).status_code == 404


def test_feedback_and_summary(client, fake):
    _, data = ask(client, "What is diabetes?")
    assert client.post(f"/api/messages/{data['message_id']}/feedback", json={"value": -1}).status_code == 200
    assert client.post(f"/api/messages/{data['message_id']}/feedback", json={"value": 5}).status_code == 400
    summary = client.get("/api/feedback/summary").get_json()
    assert summary["down"] == 1 and summary["recent_down"][0]["question"] == "What is diabetes?"


def test_health_check(client):
    assert client.get("/healthz").get_json() == {"status": "ok"}


def test_home_page_renders(client):
    page = client.get("/")
    assert page.status_code == 200 and b"Dr. Insight" in page.data
