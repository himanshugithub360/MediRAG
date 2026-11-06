from flask import Flask, jsonify, request, render_template, session
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import os

# Serve /static files at the site root too, so /favicon-32x32.png etc. resolve
app = Flask(__name__, static_folder='static', static_url_path='')
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
MISTRAL_API_KEY = os.environ.get('MISTRAL_API_KEY')
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'fallback-secret-key')

missing = [k for k, v in {'PINECONE_API_KEY': PINECONE_API_KEY, 'MISTRAL_API_KEY': MISTRAL_API_KEY}.items() if not v]
if missing:
    raise SystemExit(f"Missing environment variables: {', '.join(missing)}. Add them to your .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["MISTRAL_API_KEY"] = MISTRAL_API_KEY

index_name = "medichatbot"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

system_prompt = (
    "You are Dr. Insight, a knowledgeable and empathetic AI medical assistant. "
    "Your role is to provide accurate, evidence-based medical information based strictly on the provided context. "
    "Be concise, clear, and compassionate. If the context does not contain enough information to answer, say so honestly. "
    "Never fabricate medical facts. Always recommend consulting a real doctor for diagnosis or treatment. "
    "For emotional statements, acknowledge the person's feelings first before providing information.\n\n"
    "Medical Context:\n{context}\n\n"
    "Conversation History:\n{history}\n\n"
)

# Use the correct namespace matching store_index.py
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
    namespace="medical_data"
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

llm = ChatMistralAI(
    # Override via MISTRAL_MODEL in .env (e.g. mistral-large-latest on a paid tier)
    model=os.environ.get("MISTRAL_MODEL", "open-mistral-nemo"),
    temperature=0.2,
    api_key=MISTRAL_API_KEY
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

QUICK_RESPONSES = {
    "hi": "Hello! I'm Dr. Insight, your AI medical assistant. How can I help you today?",
    "hello": "Hi there! What health questions can I help you with?",
    "hey": "Hey! I'm here to help with your health-related questions.",
    "how are you": "I'm an AI, so I don't have feelings — but I'm fully operational and ready to assist you!",
    "thank you": "You're welcome! Feel free to ask anything else.",
    "thanks": "Happy to help! Let me know if you need anything else.",
    "bye": "Take care of yourself! Don't hesitate to come back if you have more questions.",
    "goodbye": "Goodbye! Stay healthy!",
}


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "Please enter a message."})

    lower_msg = user_message.lower()

    if "history" not in session:
        session["history"] = []

    # Quick responses for greetings/farewells
    for keyword, reply in QUICK_RESPONSES.items():
        if lower_msg == keyword or lower_msg.startswith(keyword + " ") or lower_msg.endswith(" " + keyword):
            session["history"].append(f"User: {user_message}")
            session["history"].append(f"Assistant: {reply}")
            return jsonify({"response": reply})

    conversation_history = "\n".join(session["history"][-8:])

    try:
        response = rag_chain.invoke({
            "input": user_message,
            "history": conversation_history,
        })

        bot_response = response.get("answer", "").strip()

        if not bot_response:
            bot_response = "I don't have enough information to answer that accurately. Could you rephrase, or consult a healthcare professional for this concern?"

    except Exception as e:
        app.logger.exception("Error generating response: %s", e)
        bot_response = "I'm having trouble processing that request. Please try again."

    session["history"].append(f"User: {user_message}")
    session["history"].append(f"Assistant: {bot_response}")
    session.modified = True

    return jsonify({"response": bot_response})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
