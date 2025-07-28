import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
from app.schemas import ChatRequest, ChatResponse
from app.agent import detect_lang, emergency_red_flags, call_llm

load_dotenv(dotenv_path=".env")

app = FastAPI(title="AIClinic Backend", version="0.1.0")

origins_env = os.getenv("ALLOWED_ORIGINS", "")
origins = [o.strip() for o in origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],  # relax for demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    print(os.getenv("OPENAI_API_KEY"))  # for debugging
    return {"ok": True}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_text = (req.message or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Empty message")

    lang = detect_lang(user_text)
    flag = emergency_red_flags(user_text)
    if flag:
        if lang == "ka":
            reply = (
                "თქვენი აღწერიდან არსებობს გადაუდებელი სიმპტომის რისკი ( მაგ., '{}').\n"
                "გთხოვთ, დაუყოვნებლივ დარეკოთ 112‑ზე ან მიმართოთ უახლოეს გადაუდებელ განყოფილებას.\n"
                "თუ შეგიძლიათ, თან წაიღეთ მიმდინარე მედიკამენტების სია და ალერგიების ინფორმაცია."
            ).format(flag)
        elif lang == "ru":
            reply = (
                "По описанию возможен признак неотложного состояния (напр., '{}').\n"
                "Немедленно звоните 112 или обратитесь в ближайшее отделение неотложной помощи.\n"
                "Если возможно, возьмите с собой список принимаемых препаратов и аллергий."
            ).format(flag)
        else:
            reply = (
                "Your description suggests a possible urgent symptom (e.g., '{}').\n"
                "Please call **112** or go to the nearest emergency department **now**.\n"
                "If you can, bring a list of your medications and allergies."
            ).format(flag)
        return ChatResponse(reply=reply)

    model = os.getenv("MODEL_NAME", "gpt-4o")
    reply = call_llm(user_text, model=model, lang=lang)
    return ChatResponse(reply=reply)
