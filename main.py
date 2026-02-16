import os
import json
from datetime import datetime
import time
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq
from openai import OpenAI
from google import generativeai
from groq_provider import call_groq
from gemini_provider import call_gemini
from dotenv import load_dotenv
load_dotenv()

# ---------------- CONFIG ----------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GPT_API_KEY = os.getenv("GPT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_ROUTING = {
    "LOGICA": "llama-3.3-70b-versatile",
    "TECNICO": "llama-3.3-70b-versatile",
    "DESCONOCIDO": "llama-3.3-70b-versatile"
}

GEMINI_MODELS = {
    "CREATIVO": "gemini-2.5-pro",
    "CONVERSACIONAL": "gemini-2.5-flash",
    "EDUCATIVO": "gemini-2.5-pro",
    "CONCEPTUAL": "gemini-2.5-pro",
    "DESCONOCIDO": "gemini-2.5-flash"

}

GPT_MODELS = {
    "LOGICA": "gpt-4",
    "EDUCATIVO": "gpt-4",
    "DESCONOCIDO": "gpt-4"
}
MEMORY_FILE = "memory.json"
LOG_FILE = "xy3_logs.json"

# ---------------- INIT ------------------
app = Flask(__name__)
groq_client = Groq(api_key=GROQ_API_KEY)
gpt_client = OpenAI(api_key = GPT_API_KEY)
genai_client = generativeai.configure(api_key=GEMINI_API_KEY)

# ---------------- SYSTEM PROMPT ----------------
SYSTEM_PROMPT = """
Eres XY3, una inteligencia artificial desarrollada dentro del proyecto OpenLogAI, creado por Patricio Abarca.
Tu proposito principal es asistir, guiar y razonar junto al usuario, no solo responder.

IDENTIDAD:
- Tu nombre es XY3.
- Actúas como un asistente analítico, educativo y general.
- Mantienes un tono claro, directo y respetuoso.
- No usas lenguaje innecesariamente complejo si no aporta valor.

FILOSOFÍA DE RESPUESTA:
- Prioriza el razonamiento lógico y paso a paso.
- Cuando el usuario busca aprender, no des la respuesta final inmediatamente.
- Guía al usuario mediante preguntas, pistas y descomposición del problema.
- Si el usuario pide una respuesta directa explícitamente, puedes proporcionarla.

COMPORTAMIENTO:
- Detecta errores conceptuales o lógicos y corrígelos con respeto.
- Si no tienes información suficiente o certeza, admítelo claramente.
- No inventes datos ni afirmes cosas dudosas como hechos.
- Mantén coherencia entre mensajes usando la memoria del chat.

MODO ESTUDIO:
- Fomenta la comprensión, no la memorización.
- Explica el "por qué" antes del "qué".
- Usa ejemplos simples antes de avanzar a los complejos.

MODO RAZONAMIENTO:
- Divide problemas grandes en pasos pequeños.
- Explica cada paso de forma lógica y ordenada.
- Señala supuestos y verifica conclusiones.

LIMITACIONES:
- No remplazas a un profesional humano.
- No das consejos médicos, legales o técnicos críticos como definitivos.
- En esos casos, orientas de forma general y responsable.

ACTITUD:
- Eres paciente, curioso y preciso.
- Buscas que el usuario piense mejor después de hablar contigo.
"""

# ---------------- INTENT PROMPTS ----------------
INTENT_PROMPTS = {
    "LOGICA": "Razonas paso a paso antes de responder...",
    "EDUCATIVO": "No das la respuesta directa inmediatamente...",
    "TECNICO": "Eres preciso y técnico, das ejemplos de código...",
    "CONCEPTUAL": "Defines conceptos de forma clara...",
    "CREATIVO": "Eres creativo y original...",
    "CONVERSACIONAL": "Respondes de forma natural y cercana...",
    "DESCONOCIDO": "Respondes de forma general y útil..."
}

# ---------------- MEMORIA ----------------
def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {"chats": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def summarize_messages(messages, limit=20):
    if len(messages) <= limit:
        return messages
    return messages[-limit:]

memory = load_memory()

def normalize_text(text: str) -> str:
    return text.lower().strip().replace("¿", "").replace("?", "").replace("¡", "").replace("!", "")

# ---------------- DETECCIÓN DE INTENCIONES ----------------
def detect_intent(text: str) -> str:
    t = text.lower().strip()

    has_code = any(x in t for x in ["def", "{", "}", ";", "import", "api", "error"])
    has_math = any(x in t for x in ["+", "-", "*", "/", "=", "^"])
    has_question = "?" in t
    is_long = len(t.split()) > 20

    if has_code:
        return "TECNICO"

    if has_math:
        return "LOGICA"
        
    keywords = {
        "LOGICA": ["resolver", "calcula", "ecuación", "ecuacion", "paso a paso"],
        "EDUCATIVO": ["explícame", "explicame", "ayudame", "ayúdame", "aprender", "enseña"],
        "TECNICO": ["código", "codigo", "error", "función", "funcion", "api", "python", "flask"],
        "CREATIVO": ["escribe", "inventa", "historia", "poema", "cuento"],
        "CONCEPTUAL": ["qué es", "que es", "define", "concepto", "significa"],
        "CONVERSACIONAL": ["qué opinas", "que opinas", "me siento", "hablemos", "crees que"]
    }

    for intent, words in keywords.items(): 
        for w in words:
            if w in t:
                return intent

    if has_question or is_long:
        return "EDUCATIVO"

    return "CONVERSACIONAL"

def detect_learning_mode(text: str) -> str:
    t = text.lower()

    direct_keywords = [
        "directo", "completo", "rápido", "rapido", "sin preguntas", "sin razonar", "dímelo", "dimelo", "tal cual", "tal cuál"
    ]
    for w in direct_keywords:
        if w in t:
            return "DIRECTO"

    return "GUIADO"
# ---------------- HELPERS ----------------

# ---------------- CHAT / MEMORIA ----------------
def get_or_create_chat(chat_id, first_message=""):
    chat = next((c for c in memory["chats"] if c["id"] == chat_id), None)

    if not chat:
        chat = {
            "id": chat_id or int.from_bytes(os.urandom(4), "big"),
            "title": (first_message[:40]
                       if first_message else f"Nuevo chat {len(memory['chats']) + 1}"
                     ),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "messages": []
        }
        memory["chats"].append(chat)
        
    return chat

def add_user_message(chat, content):
    chat["messages"].append({"role": "user", "content": content})
    chat["messages"] = summarize_messages(chat["messages"])

def add_assistant_message(chat, content):
    chat["messages"].append({"role": "assistant", "content": content})
    chat["messages"] = summarize_messages(chat["messages"])

# ---------------- ANALISIS ---------------

def analyze_message(text: str) -> str:
    return detect_intent(text)

# ---------------- SELECCION DE MODELO ----------------
def select_route(intent: str): 
    """Devuelve (proveedor, modelo) según el intent"""
    if intent in ("LOGICA", "EDUCATIVO"):
        return "gpt", GPT_MODELS.get(intent, GPT_MODELS["DESCONOCIDO"])

    elif intent.startswith("TECNICO",):
        return "groq", MODEL_ROUTING.get(intent, MODEL_ROUTING["DESCONOCIDO"])
   
    elif intent in ("CREATIVO", "CONVERSACIONAL", "CONCEPTUAL"):
        return "gemini", GEMINI_MODELS.get(intent, GEMINI_MODELS["DESCONOCIDO"])
    else:
        return "gemini", GEMINI_MODELS["DESCONOCIDO"]

# ---------------- LOGGING ----------------

def log_event(chat_id, intent, model, duration, status="OK", error=None):
    log_entry = { "timestamp": datetime.now().isoformat(timespec="seconds"),
                 "chat_id": chat_id,
                "intent": intent,
                "model": model,
                 "duration_sec": round(duration,3),
                "status": status
                }
    if error:
        log_entry["error"] = str(error)

    if not os.path.exists(LOG_FILE):
        logs = []
    else:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            logs = json.load(f)
            
    logs.append(log_entry)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
         json.dump(logs, f, ensure_ascii=False, indent=2)

# ---------------- ANALIZADOR DE LOGS ------------------

def analyze_logs():
    if not os.path.exists(LOG_FILE):
        return {
            "total_requests": 0,
            "by_intent": {},
            "by_model": {},
            "errors": 0,
            "avg_duration_sec": 0.0
        }

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        logs = json.load(f)

    total = len(logs)
    intents = {}
    models = {}
    errors = 0
    durations = []

    for entry in logs:
        intent = entry.get("intent") or "DESCONOCIDO"
        model = entry.get("model") or "DESCONOCIDO"
        status = entry.get("status") or "OK"
        duration = entry.get("duration_sec", 0)
        
        intents[intent] = intents.get(intent, 0) + 1
        models[model] = models.get(model, 0) + 1

        if status != "OK":
         errors += 1

        if isinstance(duration, (int, float)):
           durations.append(duration)

    avg_duration = round(sum(durations) / len(durations), 3) if durations else 0.0

    return {
        "total_requests": total,
        "by_intent": intents,
        "by_model": models,
        "errors": errors,
        "avg_duration_sec": avg_duration
    }
    

# ---------------- LLM ----------------

def call_llm(messages, intent, chat_id) -> str:
    provider, model = select_route(intent)
    provider = (provider or "").lower().strip()
    reply_text: str | None = None

    learning_mode = detect_learning_mode(messages[-1]["content"])
    system_content = SYSTEM_PROMPT + "\n" + INTENT_PROMPTS.get(intent,         INTENT_PROMPTS["DESCONOCIDO"])

    if learning_mode != "DIRECTO":
        if intent in ("LOGICA", "EDUCATIVO"):
            system_content += ("\nCuando respondas, divide la explicación en pasos claros y numerados."
                               " Haz que el usuario pueda seguir cada paso antes de continuar.")

        elif intent.startswith("TECNICO"):
            system_content += ("\nSí la pregunta involucra código o errores, explica paso a paso cada sección")
    messages = [{"role": "system", "content": system_content}, {
        "role": "system",
        "content": (
            "Antes de responder al usuario, razona internamente paso a paso."
            "No muestres ese razonamiento."
            "Entrega solo la respuesta final clara y correcta."
        )
    }] + messages

    start_time = time.time()
    duration = None

    try:
        if provider == "groq":
            reply_text = call_groq(messages, model)

        elif not reply_text:
            reply_text = call_gemini(messages, GEMINI_MODELS["DESCONOCIDO"])

        elif provider == "gemini":
            reply_text = call_gemini(messages, model)

        elif provider == "gpt":
            completion = gpt_client.chat.completions.create(
                model=model,
                messages=messages
            )
            reply_text = completion.choices[0].message.content

        else:
            raise ValueError(f"Proveedor no soportado: {provider}")

        status = "OK"
        error = None          

    except Exception as e:
        reply_text = "Oops, hubo un problema al procesar la respuesta."
        status = "ERROR"
        error = str(e)

    finally:
        duration = round(time.time() - start_time, 3)

    log_event(
        chat_id=chat_id,
        intent=intent,
        model=model,
        duration=duration,
        status=status,
        error=error 
    )
    reply_text = reply_text or ""
    if not reply_text.strip():
        reply_text = "No tengo suficiente información para responder eso correctamente. ¿ Puedes darme un poco más de contexto?"
    
    return reply_text, provider, model

#------------------ VALIDACION ------------------

def validate_response(text: str, intent: str, learning_mode: str = "GUIADO") -> str:
    text = text or ""
    if not text or len(text.strip()) < 5:
        return "No tengo suficiente información para responder correctamente. ¿Puedes reformular la pregunta?"

    if len(text) > 3000: 
        return text[:3000] + "\n\n[Respuesta recortada por longitud]"

    if intent == "LOGICA" and "paso a paso" not in text.lower():
        return "Vamos a resolverlo paso a paso:\n\n" + text

    if intent == "EDUCATIVO" and "?" not in text:
        return text + "\n\n¿Quieres que lo veamos con un ejemplo?"

    if learning_mode != "DIRECTO" and intent in ("LOGICA", "EDUCATIVO"):
        if "paso" not in text.lower():
            text = "Vamos a resolverlo paso a paso:\n\n" + text

    return text

def self_check(text: str, intent: str) -> str:

    if intent in ("LOGICA", "TECNICO"):
        if "no sé" in text.lower() or "no estoy seguro" in text.lower():
            return "Voy a analizarlo con más cuidado.\n\n" + text

    if not text or len(text.strip()) < 5: 
        text = "No tengo suficiente información para responder correctamente. ¿Puedes reformular la pregunta?"

    text = normalize_text(text)
    
    return text
# ---------------- MANEJO DE CHATS ----------------

@app.route("/chat/new", methods=["POST"])
def new_chat():
    """Crea un nuevo chat vacío o con primer mensaje opcional"""
    data = request.json
    first_message = data.get("first_message", "")
    chat = get_or_create_chat(None, first_message=first_message)
    save_memory(memory)
    return jsonify({"chat_id": chat["id"], "message": first_message})

@app.route("/chat/delete", methods=["POST"])
def delete_chat():
    data = request.json
    chat_id = data.get("chat_id")
    if not chat_id:
        return jsonify({"error": "No se proporcionó chat_id"}), 400

    global memory
    memory["chats"] = [c for c in memory["chats"] if c["id"] != chat_id]
    save_memory(memory)
    return jsonify({"status": "ok", "chat_id": chat_id})
# ---------------- RUTAS ----------------
@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat_route():
    global memory
    memory = load_memory()
    data = request.json
    raw_text = data.get("message", "").strip()
    chat_id = data.get("chat_id")

    if not raw_text:
        return jsonify({"reply": "Mensaje vacío"}), 400

    normalized_text = normalize_text(raw_text)

    chat = get_or_create_chat(chat_id, first_message=raw_text)
    add_user_message(chat, raw_text)
    
    intent = detect_intent(normalized_text)
    print(f"[INTENT] Detectado: {intent}")
    
    try:
        reply_text, provider, model = call_llm(chat["messages"], intent, chat["id"])
        reply_text = validate_response(reply_text or "", intent)
        reply_text = self_check(reply_text, intent)
    
    except Exception as e:
        print("ERROR LLM:", e)
        reply_text = "Oops, hubo un problema al procesar la respuesta."

    add_assistant_message(chat, reply_text)
    save_memory(memory)


    return jsonify({
        "reply": reply_text, 
        "chat_id": chat["id"],
        "provider": provider,
        "model": model})

@app.route("/chats", methods=["GET"])
def list_chats():
    global memory
    memory = load_memory()
    
    chats_summary = [{
        "id": c["id"],
        "title": c.get("title", "Chat"),
        "created_at": c.get("created_at", ""),
        "last_message": (c["messages"][-1]["content"]
                         if c["messages"] else ""
        )
    }
                     for c in memory["chats"]
    ]

    return jsonify(chats_summary)

@app.route("/chat/<int:chat_id>", methods=["GET"])
def get_chat(chat_id):
    global memory
    memory = load_memory()
    chat = next((c for c in memory["chats"] if c["id"] == chat_id), None)
    if not chat:
        return jsonify({"error": "Chat no encontrado"}), 404
    return jsonify(chat)


# ---------------- RUN -------------------
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 5000))
    print(f"Servidor XY3 corriendo en http://127.0.0.1:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=True)
