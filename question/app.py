# app.py - Enhanced MCQ Generator with Groq API
import json
import re
from typing import Dict, Any, List
from fastapi import FastAPI, Query, HTTPException, Body
from pydantic import BaseModel
import uvicorn
import requests
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

app = FastAPI(title="MCQ Generator API", version="1.0.0")

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ✅ Fixed CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080","*"],  # allow all origins for now (change in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request models
class QuizRequest(BaseModel):
    topic: str
    domain: str
    difficulty: str = "intermediate"  # beginner, intermediate, advanced
    num_questions: int = 10
    focus_areas: List[str] = []  # specific skills to focus on
    user_level: str = "intermediate"

class MCQGeneratorAgent:
    def __init__(self):
        self.model = "llama-3.1-8b-instant"  # You can use "llama3-8b-8192" for faster responses

    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API with the given prompt"""
        if not GROQ_API_KEY:
            raise Exception("Groq API key not configured")
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 6000,
            "top_p": 0.9,
            "stream": False
        }
        
        try:
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    def generate_mcqs(self, request: QuizRequest) -> Dict[str, Any]:
        prompt = f"""
        You are an expert educational content creator. Generate {request.num_questions} multiple-choice questions
        for a skills assessment with the following specifications:

        Topic: "{request.topic}"
        Domain: "{request.domain}"
        Difficulty Level: {request.difficulty}
        User Level: {request.user_level}
        Focus Areas: {', '.join(request.focus_areas) if request.focus_areas else 'General skills'}

        Requirements:
        1. Exactly {request.num_questions} MCQs
        2. Each question should have 4 options (A, B, C, D)
        3. Questions should be appropriate for {request.difficulty} level
        4. Include practical, real-world scenarios
        5. Provide correct answer and detailed explanation for learning
        6. Return ONLY valid JSON in this exact format:

        {{
            "quiz_metadata": {{
                "topic": "{request.topic}",
                "domain": "{request.domain}",
                "difficulty": "{request.difficulty}",
                "total_questions": {request.num_questions},
                "estimated_time": "15-20 minutes"
            }},
            "questions": [
                {{
                    "id": 1,
                    "question": "Question text here",
                    "options": {{
                        "A": "Option A text",
                        "B": "Option B text",
                        "C": "Option C text",
                        "D": "Option D text"
                    }},
                    "correct_answer": "A",
                    "explanation": "Detailed explanation of why this is correct",
                    "skill_category": "Technical Skills",
                    "difficulty_score": 7
                }}
            ]
        }}

        Important: Return ONLY valid JSON, no other text or markdown formatting.
        """

        try:
            raw_text = self._call_groq_api(prompt)

            # ✅ Clean unwanted markdown fences (```json ... ```)
            clean_text = re.sub(r"^```(?:json)?", "", raw_text)
            clean_text = re.sub(r"```$", "", clean_text).strip()

            # Parse JSON
            quiz_data = json.loads(clean_text)
            
            # Validate the structure
            if "questions" not in quiz_data:
                raise ValueError("Generated JSON missing 'questions' field")
                
            if len(quiz_data["questions"]) != request.num_questions:
                print(f"Warning: Requested {request.num_questions} questions, got {len(quiz_data['questions'])}")

            return quiz_data

        except json.JSONDecodeError as je:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse generated quiz JSON: {str(je)}\nRaw response: {raw_text[:500]}..."
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Quiz generation failed: {str(e)}"
            )

# ---------------- Endpoints ----------------
@app.get("/")
def root():
    return {
        "message": "MCQ Generator API for Skills Assessment 🚀",
        "version": "1.0.0",
        "ai_provider": "Groq API",
        "model": "llama3-70b-8192",
        "endpoints": {
            "/": "Base endpoint",
            "/generate": "Generate MCQs (GET with query params)",
            "/generate-quiz": "Generate MCQs (POST with detailed request)",
            "/health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    try:
        # Test Groq API connection
        agent = MCQGeneratorAgent()
        test_prompt = "Say 'Hello World'"
        response = agent._call_groq_api(test_prompt)
        
        return {
            "status": "healthy", 
            "service": "MCQ Generator",
            "ai_provider": "Groq API",
            "groq_connection": "working"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "MCQ Generator",
            "ai_provider": "Groq API", 
            "groq_connection": "failed",
            "error": str(e)
        }

@app.get("/generate")
def generate_legacy(
    topic: str = Query(..., description="Quiz topic"),
    domain: str = Query(..., description="Subject domain"),
    num_questions: int = Query(10, description="Number of questions"),
    difficulty: str = Query("intermediate", description="Difficulty level")
):
    agent = MCQGeneratorAgent()
    request = QuizRequest(
        topic=topic,
        domain=domain,
        num_questions=num_questions,
        difficulty=difficulty
    )
    return agent.generate_mcqs(request)

@app.post("/generate-quiz")
def generate_quiz_detailed(request: QuizRequest):
    """Generate MCQs with detailed request body"""
    agent = MCQGeneratorAgent()
    return agent.generate_mcqs(request)

@app.post("/generate")
def generate_legacy_post(
    topic: str = Query(..., description="Quiz topic"),
    domain: str = Query(..., description="Subject domain"),
    num_questions: int = Query(10, description="Number of questions")
):
    agent = MCQGeneratorAgent()
    request = QuizRequest(
        topic=topic,
        domain=domain,
        num_questions=num_questions
    )
    return agent.generate_mcqs(request)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)