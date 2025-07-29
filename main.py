import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger
from app.schemas import ChatRequest, ChatResponse, SymptomAssessment, SymptomAssessment
from app.agent import detect_lang, emergency_red_flags, call_llm, get_disease_guidance, generate_suggested_actions

load_dotenv(dotenv_path=".env")

app = FastAPI(title="AIClinic Backend", version="0.1.0")

origins_env = os.getenv("ALLOWED_ORIGINS", "")
if origins_env:
    origins = [o.strip() for o in origins_env.split(",")]
else:
    origins = None
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
        return ChatResponse(
            reply=reply,
            urgency_level="emergency",
            suggested_actions=["Call 112", "Go to emergency department", "Bring medication list"]
        )

    model = os.getenv("MODEL_NAME", "gpt-4o")
    response_data = call_llm(user_text, model=model, lang=lang, conversation_context=req.user_context)
    
    return ChatResponse(
        reply=response_data["reply"],
        follow_up_questions=response_data.get("follow_up_questions"),
        suggested_actions=response_data.get("suggested_actions"),
        urgency_level=response_data.get("urgency_level"),
        disease_info=response_data.get("disease_info")
    )

@app.post("/api/symptom-assessment")
def assess_symptom(assessment: SymptomAssessment):
    """Enhanced endpoint for structured symptom assessment"""
    try:
        lang = detect_lang(assessment.symptom)
        disease_info = get_disease_guidance(assessment.symptom.lower(), lang)
        suggested_actions = generate_suggested_actions(assessment.symptom.lower(), "low", lang)
        
        # Determine urgency based on severity
        urgency_level = "low"
        if assessment.severity >= 8:
            urgency_level = "high"
        elif assessment.severity >= 5:
            urgency_level = "moderate"
        
        return {
            "symptom": assessment.symptom,
            "severity_level": urgency_level,
            "disease_info": disease_info,
            "suggested_actions": suggested_actions,
            "recommendations": {
                "seek_immediate_care": assessment.severity >= 8,
                "monitor_symptoms": True,
                "follow_up_days": 3 if assessment.severity < 5 else 1
            }
        }
    except Exception as e:
        logger.exception("Symptom assessment error")
        raise HTTPException(status_code=500, detail="Failed to assess symptom")

@app.get("/api/disease-guidelines/{condition}")
def get_disease_guidelines(condition: str, lang: str = "en"):
    """Get specific disease guidelines"""
    try:
        from app.agent import DISEASE_GUIDELINES
        
        if condition.lower() in DISEASE_GUIDELINES:
            guidelines = DISEASE_GUIDELINES[condition.lower()]
            return {
                "condition": condition,
                "guidance": guidelines["guidance"].get(lang, guidelines["guidance"]["en"]),
                "red_flags": guidelines["red_flags"],
                "symptoms": guidelines["symptoms"]
            }
        else:
            raise HTTPException(status_code=404, detail="Condition not found")
    except Exception as e:
        logger.exception("Guidelines fetch error")
        raise HTTPException(status_code=500, detail="Failed to fetch guidelines")
