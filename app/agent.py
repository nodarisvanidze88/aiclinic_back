import os
from typing import Optional
from loguru import logger
import dotenv

dotenv.load_dotenv(dotenv_path="env")

# Simple language "guess" – very naive, just for demo
def detect_lang(text: str) -> str:
    ka_chars = sum(ch >= "\u10A0" and ch <= "\u10FF" for ch in text)
    ru_chars = sum("\u0400" <= ch <= "\u04FF" for ch in text)
    if ka_chars > ru_chars and ka_chars > 0:
        return "ka"
    if ru_chars > 0:
        return "ru"
    return "en"

def emergency_red_flags(text: str) -> Optional[str]:
    flags = [
        # EN
        ("chest pain", "en"), ("shortness of breath", "en"), ("severe headache", "en"),
        ("loss of consciousness", "en"), ("stroke", "en"), ("suicidal", "en"),
        # KA
        ("მკერდის ტკივილი", "ka"), ("სუნთქვის უკმარისობა", "ka"), ("ძლიერი თავის ტკივილი", "ka"),
        ("ბოდავს", "ka"), ("თვითმკვლელობა", "ka"),
        # RU
        ("боль в груди", "ru"), ("одышка", "ru"), ("сильная головная боль", "ru"),
        ("потеря сознания", "ru"), ("инсульт", "ru"), ("суицид", "ru"),
    ]
    lower = text.lower()
    for kw, _ in flags:
        if kw in lower:
            return kw
    return None

SYSTEM_PROMPT = """
You are a careful, upbeat **virtual primary-care triage assistant**.
**You are not a doctor and do not provide diagnoses or prescriptions.**
Your role:
- Gather concise symptom history (onset, duration, severity, triggers, meds, allergies, relevant conditions).
- Give next-step guidance (self-care, OTC where appropriate, when to seek in-person care).
- Avoid fear-inducing language. Be short and clear.
- If an emergency is suspected, direct to **local emergency services immediately**.
- Never prescribe controlled or prescription-only drugs.
- Be available in **Georgian**, **English**, or **Russian**. Answer in the user's language and keep a warm, reassuring tone.

Formatting:
- Use short paragraphs or bullet points.
- End with a short checklist of what to monitor and when to seek urgent care.
"""

# ---- OpenAI client (lazy import so app starts without key in dev) ----
def call_llm(prompt_user: str, model: str, lang: str) -> str:
    print(os.getenv("OPENAI_API_KEY"))
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # system + user messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Prefix instruction per language
        if lang == "ka":
            messages.append({"role": "system", "content":
                "Respond in Georgian. Keep a friendly, calm style. Keep it brief."})
        elif lang == "ru":
            messages.append({"role": "system", "content":
                "Respond in Russian. Keep a friendly, calm style. Keep it brief."})
        else:
            messages.append({"role": "system", "content":
                "Respond in English. Keep a friendly, calm style. Keep it brief."})

        messages.append({"role": "user", "content": prompt_user})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.exception("LLM error")
        # Fallback generic response
        if lang == "ka":
            return ("ვწუხვარ, პასუხის გენერაცია ვერ გამოვიდა. სცადეთ მოგვიანებით.\n"
                    "ამასობაში, თუ სიმპტომები მძიმდება ან გაქვთ გადაუდებელი მდგომარეობა, "
                    "დაეკონტაქტთ 112‑ს.")
        elif lang == "ru":
            return ("Извините, не удалось получить ответ. Попробуйте позже.\n"
                    "Если состояние ухудшается или это неотложная ситуация — звоните 112.")
        else:
            return ("Sorry, I couldn’t generate a response right now. Please try again later.\n"
                    "If your symptoms worsen or you suspect an emergency, call local emergency services (112).")
