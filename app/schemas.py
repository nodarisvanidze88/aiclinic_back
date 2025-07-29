from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_context: Optional[dict] = None  # age, gender, medical_history, etc.

class ChatResponse(BaseModel):
    reply: str
    follow_up_questions: Optional[List[str]] = None
    suggested_actions: Optional[List[str]] = None
    urgency_level: Optional[str] = None  # "low", "moderate", "high", "emergency"
    disease_info: Optional[dict] = None

class SymptomAssessment(BaseModel):
    symptom: str
    severity: int  # 1-10 scale
    duration: str
    triggers: Optional[List[str]] = None
    associated_symptoms: Optional[List[str]] = None
