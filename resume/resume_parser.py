# resume_parser.py
import pdfplumber
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_groq_api(prompt: str, model: str = "llama-3.1-8b-instant") -> str:
    """Call Groq API with the given prompt"""
    if not GROQ_API_KEY:
        raise Exception("Groq API key not configured")
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": model,
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stream": False
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")

def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber"""
    text = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")

def ask_llm(prompt: str) -> str:
    """Send prompt to Groq LLM"""
    return call_groq_api(prompt)

def clean_json_response(response: str) -> dict:
    """Clean and parse JSON response from LLM"""
    try:
        # Remove markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        elif response.startswith("```"):
            response = response[3:]
        
        if response.endswith("```"):
            response = response[:-3]
        
        return json.loads(response.strip())
    except json.JSONDecodeError as e:
        raise Exception(f"Failed to parse JSON response: {e}\nResponse was: {response}")

def get_parse_prompt(resume_text: str) -> str:
    return f"""
You are an expert resume parser.
Extract structured data from the following resume and return valid JSON ONLY.

Required fields:
- "name" (string)
- "contact" (object with email, phone, linkedin)
- "skills" (array of strings)
- "experience" (array of objects with title, company, start_date, end_date, description)
- "education" (array of objects with degree, field, institution, year)
- "projects" (array of strings)
- "certifications" (array of strings)

Important: Return ONLY valid JSON, no other text.

Resume:
{resume_text[:3000]}  # Limit text to avoid token limits
"""

def get_score_prompt(resume_json: str) -> str:
    return f"""
You are an Applicant Tracking System (ATS).

Resume JSON:
{resume_json}

Score the resume (0–100) with these weights:
- Skills match (40%)
- Experience relevance (20%)
- Title/role alignment (15%)
- Education/certs (10%)
- Formatting/parseability (10%)
- Language/grammar (5%)

Return JSON ONLY with these exact fields:
{{
  "skill_score": number,
  "experience_score": number,
  "title_score": number,
  "education_score": number,
  "format_score": number,
  "language_score": number,
  "total_score": number
}}

Important: Return ONLY valid JSON, no other text.
"""

def get_recommend_prompt(resume_json: str) -> str:
    return f"""
You are a professional resume coach.

Resume JSON: {resume_json}

Return JSON ONLY with these exact fields:
{{
  "missing_skills": array of strings,
  "improved_bullets": array of strings,
  "recommendations": array of strings,
  "summary": string
}}

Important: Return ONLY valid JSON, no other text.
"""

def parse_resume(resume_text: str) -> dict:
    """Parse resume text into structured JSON"""
    prompt = get_parse_prompt(resume_text)
    response = ask_llm(prompt)
    return clean_json_response(response)

def score_resume(resume_json: dict) -> dict:
    """Score resume for ATS compatibility"""
    prompt = get_score_prompt(json.dumps(resume_json, indent=2))
    response = ask_llm(prompt)
    return clean_json_response(response)

def recommend_improvements(resume_json: dict) -> dict:
    """Get recommendations for resume improvement"""
    prompt = get_recommend_prompt(json.dumps(resume_json, indent=2))
    response = ask_llm(prompt)
    return clean_json_response(response)

def analyze_resume_pdf(pdf_path: str) -> dict:
    """
    Complete resume analysis pipeline:
    1. Extract text from PDF
    2. Parse into structured data
    3. Score for ATS
    4. Get improvement recommendations
    """
    try:
        # Step 1: Extract text
        print("📄 Extracting text from PDF...")
        resume_text = extract_text_from_pdf(pdf_path)
        
        if not resume_text.strip():
            raise Exception("No text could be extracted from the PDF")
        
        # Step 2: Parse resume
        print("🔍 Parsing resume structure...")
        parsed_resume = parse_resume(resume_text)
        
        # Step 3: Score resume
        print("📊 Scoring resume for ATS...")
        scores = score_resume(parsed_resume)
        
        # Step 4: Get recommendations
        print("💡 Generating recommendations...")
        recommendations = recommend_improvements(parsed_resume)
        
        return {
            "success": True,
            "parsed_resume": parsed_resume,
            "scores": scores,
            "recommendations": recommendations,
            "text_preview": resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Example usage
if __name__ == "__main__":
    # Test the parser
    pdf_path = "sample_resume.pdf"  # Replace with your PDF path
    result = analyze_resume_pdf(pdf_path)
    print(json.dumps(result, indent=2))