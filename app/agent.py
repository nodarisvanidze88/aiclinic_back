import os
import json
from typing import Optional, Dict, List, Tuple
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
        ("can't breathe", "en"), ("heart attack", "en"), ("severe bleeding", "en"),
        # KA
        ("მკერდის ტკივილი", "ka"), ("სუნთქვის უკმარისობა", "ka"), ("ძლიერი თავის ტკივილი", "ka"),
        ("ბოდავს", "ka"), ("თვითმკვლელობა", "ka"), ("გულის შეტევა", "ka"),
        # RU
        ("боль в груди", "ru"), ("одышка", "ru"), ("сильная головная боль", "ru"),
        ("потеря сознания", "ru"), ("инсульт", "ru"), ("суицид", "ru"), ("инфаркт", "ru"),
    ]
    lower = text.lower()
    for kw, _ in flags:
        if kw in lower:
            return kw
    return None

# Disease-specific knowledge base
DISEASE_GUIDELINES = {
    "common_cold": {
        "symptoms": ["runny nose", "sneezing", "mild cough", "sore throat"],
        "guidance": {
            "en": "Rest, fluids, OTC pain relievers. Usually resolves in 7-10 days.",
            "ka": "დასვენება, ბევრი სითხე, ტკივილგამათლმშრალებელი. ჩვეულებრივ 7-10 დღეში გადის.",
            "ru": "Покой, обильное питье, безрецептурные обезболивающие. Обычно проходит за 7-10 дней."
        },
        "red_flags": ["high fever >38.5°C", "difficulty breathing", "severe headache"]
    },
    "flu": {
        "symptoms": ["fever", "body aches", "fatigue", "dry cough"],
        "guidance": {
            "en": "Rest, fluids, fever reducers. Antiviral drugs may help if started early.",
            "ka": "დასვენება, სითხეები, ცხელების შემამცირებელი. ანტივირუსული ადრე დაწყებით შეიძლება დაეხმაროს.",
            "ru": "Покой, жидкости, жаропонижающие. Противовирусные препараты могут помочь при раннем начале."
        },
        "red_flags": ["difficulty breathing", "persistent high fever", "severe dehydration"]
    },
    "headache": {
        "symptoms": ["head pain", "tension", "migraine"],
        "guidance": {
            "en": "Rest in dark room, hydration, OTC pain relievers. Identify triggers.",
            "ka": "დასვენება მუქ ოთახში, ჰიდრატაცია, ტკივილგამათისებელი. გამომწვევი ფაქტორების იდენტიფიცირება.",
            "ru": "Отдых в темной комнате, увлажнение, безрецептурные обезболивающие. Выявление триггеров."
        },
        "red_flags": ["sudden severe headache", "fever with headache", "vision changes"]
    },
    "stomach_pain": {
        "symptoms": ["stomach ache", "abdominal pain", "nausea", "indigestion"],
        "guidance": {
            "en": "Light diet, clear fluids, rest. Avoid dairy and fatty foods.",
            "ka": "მსუბუქი დიეტა, გამჭვირვალე სითხეები, დასვენება. ავარიდეთ რძის პროდუქტები და ცხიმიანი საკვები.",
            "ru": "Легкая диета, прозрачные жидкости, покой. Избегайте молочных и жирных продуктов."
        },
        "red_flags": ["severe pain", "vomiting blood", "high fever", "severe dehydration"]
    }
}

IMPROVED_SYSTEM_PROMPT = """
You are an AI health triage assistant. Your goal is to provide helpful, focused guidance while being concise and not overwhelming.

CORE PRINCIPLES:
1. **Ask ONE focused question at a time** - never overwhelm with multiple questions
2. **Provide structured, actionable responses**
3. **Match the user's language** (Georgian, English, or Russian)
4. **Use a warm, reassuring tone**
5. **Always include when to seek urgent care**

RESPONSE FORMAT:
- Start with empathetic acknowledgment
- Ask ONE specific follow-up question if needed
- Provide clear, actionable guidance
- End with monitoring instructions

EXAMPLE GOOD RESPONSE:
"I understand you're experiencing headaches. To help you better, what would you rate the pain on a scale of 1-10?

For headaches, try:
• Rest in a quiet, dark room
• Stay hydrated
• Consider over-the-counter pain relief

🚨 Seek immediate care if you experience:
• Sudden, severe headache
• Fever with headache
• Vision changes"

AVOID:
- Multiple questions in one response
- Medical jargon without explanation
- Lengthy paragraphs
- Diagnostic statements

Remember: You cannot diagnose or prescribe. Guide users to appropriate care levels.
"""

def analyze_symptoms(text: str, lang: str) -> Tuple[str, List[str], str]:
    """
    Analyze user input to extract symptoms and suggest relevant disease info
    Returns: (primary_symptom, related_symptoms, urgency_level)
    """
    # Simple keyword matching for demo - could be enhanced with NLP
    symptom_keywords = {
        "headache": ["headache", "head pain", "migraine", "თავის ტკივილი", "головная боль"],
        "fever": ["fever", "hot", "temperature", "ცხელება", "лихорадка"],
        "cough": ["cough", "coughing", "ხველა", "кашель"],
        "cold": ["cold", "runny nose", "sneezing", "ცივი", "простуда"],
        "stomach_pain": ["stomach", "belly", "abdominal", "nausea", "კუჭი", "живот", "тошнота"],
    }
    
    text_lower = text.lower()
    found_symptoms = []
    
    for condition, keywords in symptom_keywords.items():
        if any(kw in text_lower for kw in keywords):
            found_symptoms.append(condition)
    
    primary_symptom = found_symptoms[0] if found_symptoms else "general"
    urgency_level = "low"
    
    # Check for high urgency keywords
    high_urgency_keywords = ["severe", "intense", "can't", "unable", "emergency", "ძლიერი", "сильный"]
    if any(kw in text_lower for kw in high_urgency_keywords):
        urgency_level = "moderate"
    
    return primary_symptom, found_symptoms, urgency_level

def get_disease_guidance(symptom: str, lang: str) -> Optional[Dict]:
    """Get specific guidance for a symptom/condition"""
    for condition, info in DISEASE_GUIDELINES.items():
        if symptom in info["symptoms"] or symptom == condition:
            return {
                "condition": condition,
                "guidance": info["guidance"].get(lang, info["guidance"]["en"]),
                "red_flags": info["red_flags"]
            }
    return None

def generate_follow_up_questions(symptom: str, lang: str) -> List[str]:
    """Generate relevant follow-up questions based on symptom"""
    questions = {
        "headache": {
            "en": ["How would you rate the pain on a scale of 1-10?"],
            "ka": ["როგორ შეაფასებდით ტკივილს 1-10 შკალაზე?"],
            "ru": ["Как бы вы оценили боль по шкале от 1 до 10?"]
        },
        "fever": {
            "en": ["What's your current temperature?"],
            "ka": ["რა არის თქვენი ახლანდელი ტემპერატურა?"],
            "ru": ["Какая у вас сейчас температура?"]
        },
        "cough": {
            "en": ["Is the cough dry or with phlegm?"],
            "ka": ["ხველა მშრალია თუ კავებით?"],
            "ru": ["Кашель сухой или с мокротой?"]
        },
        "stomach_pain": {
            "en": ["When did the stomach pain start?"],
            "ka": ["როდის დაიწყო კუჭის ტკივილი?"],
            "ru": ["Когда началась боль в животе?"]
        }
    }
    
    return questions.get(symptom, {}).get(lang, questions.get(symptom, {}).get("en", []))

def generate_suggested_actions(symptom: str, urgency_level: str, lang: str) -> List[str]:
    """Generate actionable suggestions based on symptom and urgency"""
    actions = {
        "headache": {
            "en": ["Rest in a quiet, dark room", "Stay hydrated", "Consider over-the-counter pain relief"],
            "ka": ["დაისვენეთ მშვიდ, მუქ ოთახში", "დარჩით ჰიდრატებული", "განიხილეთ ტკივილგამათისებელი"],
            "ru": ["Отдохните в тихой, темной комнате", "Пейте больше жидкости", "Рассмотрите безрецептурные обезболивающие"]
        },
        "fever": {
            "en": ["Rest and stay hydrated", "Monitor temperature regularly", "Use fever reducers if needed"],
            "ka": ["დაისვენეთ და დარჩით ჰიდრატებული", "რეგულარულად აკონტროლეთ ტემპერატურა", "საჭიროების შემთხვევაში გამოიყენეთ ცხელების შემამცირებელი"],
            "ru": ["Отдыхайте и пейте много жидкости", "Регулярно измеряйте температуру", "При необходимости используйте жаропонижающие"]
        },
        "stomach_pain": {
            "en": ["Eat light foods", "Stay hydrated with clear fluids", "Rest and avoid heavy meals"],
            "ka": ["მიირთვით მსუბუქი საკვები", "დარჩით ჰიდრატებული გამჭვირვალე სითხეებით", "დაისვენეთ და ავარიდეთ მძიმე საკვები"],
            "ru": ["Ешьте легкую пищу", "Пейте прозрачные жидкости", "Отдыхайте и избегайте тяжелой пищи"]
        }
    }
    
    default_actions = {
        "en": ["Monitor symptoms", "Rest and stay hydrated", "Seek medical care if worsening"],
        "ka": ["დააკვირდით სიმპტომებს", "დაისვენეთ და დარჩით ჰიდრატებული", "მიმართეთ ექიმს თუ უარესდება"],
        "ru": ["Следите за симптомами", "Отдыхайте и пейте жидкость", "Обратитесь к врачу при ухудшении"]
    }
    
    return actions.get(symptom, {}).get(lang, default_actions.get(lang, default_actions["en"]))

# ---- OpenAI client (lazy import so app starts without key in dev) ----
def call_llm(prompt_user: str, model: str, lang: str, conversation_context: Optional[Dict] = None) -> Dict:
    print(os.getenv("OPENAI_API_KEY"))
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Analyze symptoms first
        primary_symptom, related_symptoms, urgency_level = analyze_symptoms(prompt_user, lang)
        disease_info = get_disease_guidance(primary_symptom, lang)
        
        # Enhanced system prompt with conversation context
        messages = [
            {"role": "system", "content": IMPROVED_SYSTEM_PROMPT},
        ]

        # Add language-specific instructions
        lang_instructions = {
            "ka": "უპასუხე ქართულად. იყავი მეგობრული და მშვიდი. მოკლედ და გარკვევით.",
            "ru": "Отвечай на русском языке. Будь дружелюбным и спокойным. Кратко и ясно.",
            "en": "Respond in English. Be friendly and calm. Keep it brief and clear."
        }
        
        messages.append({"role": "system", "content": lang_instructions.get(lang, lang_instructions["en"])})
        
        # Add conversation context if available
        if conversation_context:
            context_msg = f"Previous context: {json.dumps(conversation_context, ensure_ascii=False)}"
            messages.append({"role": "system", "content": context_msg})
        
        # Add disease-specific guidance if found
        if disease_info:
            guidance_msg = f"Relevant condition guidance: {disease_info['guidance']}"
            messages.append({"role": "system", "content": guidance_msg})

        messages.append({"role": "user", "content": prompt_user})

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,  # Lower temperature for more consistent medical responses
            max_tokens=300,   # Reduced to encourage concise responses
        )
        
        response_text = resp.choices[0].message.content.strip()
        
        # Generate follow-up questions based on symptom
        follow_up_questions = generate_follow_up_questions(primary_symptom, lang)
        
        # Generate suggested actions
        suggested_actions = generate_suggested_actions(primary_symptom, urgency_level, lang)
        
        return {
            "reply": response_text,
            "follow_up_questions": follow_up_questions[:1],  # Only one question at a time
            "suggested_actions": suggested_actions,
            "urgency_level": urgency_level,
            "disease_info": disease_info
        }
        
    except Exception as e:
        logger.exception("LLM error")
        # Fallback generic response
        fallback_responses = {
            "ka": "ვწუხვარ, პასუხის გენერაცია ვერ გამოვიდა. სცადეთ მოგვიანებით.\nამასობაში, თუ სიმპტომები მძიმდება, დაეკონტაქტეთ 112‑ს.",
            "ru": "Извините, не удалось получить ответ. Попробуйте позже.\nЕсли состояние ухудшается — звоните 112.",
            "en": "Sorry, I couldn't generate a response right now. Please try again later.\nIf symptoms worsen, call emergency services."
        }
        
        return {
            "reply": fallback_responses.get(lang, fallback_responses["en"]),
            "follow_up_questions": [],
            "suggested_actions": [],
            "urgency_level": "low",
            "disease_info": None
        }
