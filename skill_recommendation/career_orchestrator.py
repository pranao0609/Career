import json
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================================================
# Career Orchestrator (Groq API)
# ==================================================
class CareerOrchestrator:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.groq_api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.1-8b-instant"
        
    def _call_groq_api(self, prompt: str) -> str:
        """Call Groq API with the given prompt"""
        if not self.groq_api_key:
            raise Exception("Groq API key not configured")
            
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9,
            "stream": False
        }
        
        print(f"📤 Sending request to Groq API...")
        
        try:
            response = requests.post(self.groq_api_url, headers=headers, json=payload, timeout=30)
            print(f"📥 Response status: {response.status_code}")
            
            if response.status_code == 401:
                raise Exception("Invalid API key - Please check your GROQ_API_KEY")
            elif response.status_code == 429:
                raise Exception("Rate limit exceeded - Try again later")
            elif response.status_code != 200:
                raise Exception(f"API error {response.status_code}: {response.text}")
                
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Groq API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

    # ... rest of your methods remain the same ...
    def _parse_student_profile(self, text: str) -> dict:
        prompt = f"""
        Extract key information from the following student profile and return ONLY valid JSON.
        Required keys: 
        - "skills" (list of strings) 
        - "academics" (string) 
        - "interests" (list of strings)

        Profile Text: {text}

        Return ONLY JSON, no other text.
        """
        response = self._call_groq_api(prompt)
        try:
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except Exception as e:
            print(f"Error parsing profile JSON: {e}")
            return {"skills": [], "academics": None, "interests": []}

    def _generate_career_suggestions(self, student_profile_text: str) -> list:
        prompt = f"""
        You are a career counselor. Based on the student's profile below, suggest 3 possible career paths. 
        For each career, provide:
        - "career_name"
        - "required_skills" (list of 5–7 key skills)
        - "reasoning" (why this fits the student)

        Respond strictly in JSON array format.

        Profile: {student_profile_text}

        Return ONLY JSON array, no other text.
        """
        response = self._call_groq_api(prompt)
        try:
            # Clean the response to ensure it's valid JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            return json.loads(response.strip())
        except Exception as e:
            print(f"Error parsing career suggestions JSON: {e}")
            return [
                {
                    "career_name": "General Career Suggestion", 
                    "required_skills": ["Communication", "Problem-solving", "Adaptability"], 
                    "reasoning": "Based on your profile, these foundational skills will help in many career paths."
                }
            ]

    def _generate_explanation(self, recommendation: dict, student_profile: dict) -> str:
        prompt = f"""
        You are a career counselor. Generate a personalized explanation for the following career recommendation.

        Recommendation: {recommendation['career_name']}
        Student profile strengths: {', '.join(student_profile.get('skills', []))}
        Suggested required skills: {', '.join(recommendation.get('required_skills', []))}

        Provide a concise, encouraging explanation highlighting the alignment 
        and suggesting which skills to improve.

        Keep it to 2-3 sentences maximum.
        """
        return self._call_groq_api(prompt)

    def run(self, student_profile_text: str):
        # Step 1: Parse profile
        parsed_profile = self._parse_student_profile(student_profile_text)

        # Step 2: Get career suggestions from Groq
        suggestions = self._generate_career_suggestions(student_profile_text)

        # Step 3: Add explanations
        for s in suggestions:
            s["explanation"] = self._generate_explanation(s, parsed_profile)

        return suggestions


# ==================================================
# Resume Parser
# ==================================================
import textract

class ResumeParser:
    """
    Extracts text from resume files (.pdf, .docx, etc.)
    """
    def extract(self, file_path: str) -> str:
        try:
            text = textract.process(file_path).decode("utf-8")
            return text.strip()
        except Exception as e:
            return f"Error extracting resume: {e}"