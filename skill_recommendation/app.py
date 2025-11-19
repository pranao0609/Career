from fastapi import FastAPI, UploadFile, File
from career_orchestrator import CareerOrchestrator, ResumeParser

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://localhost:8081",
        "https://frontend-app-278398219986.asia-south1.run.app",  # Your frontend URL
        "https://*.run.app",  # Allow all Cloud Run domains
        "*"  # Allow all origins for development - remove in production
    ],
    allow_credentials=False,  # Set to False for broader compatibility
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Accept",
        "Accept-Language", 
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-Requested-With",
        "Origin",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers"
    ],
    expose_headers=["*"],
)
# Initialize orchestrator with your project + region
orchestrator = CareerOrchestrator()
resume_parser = ResumeParser()


# -------------------------
# Health Check
# -------------------------
@app.get("/")
def health_check():
    return {"status": "Career Advisor API is running!"}


# -------------------------
# Upload Resume & Get Suggestions
# -------------------------
@app.post("/analyze-resume/")
async def analyze_resume(file: UploadFile = File(...)):
    try:
        # Save uploaded resume temporarily
        file_location = f"/tmp/{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Extract text
        resume_text = resume_parser.extract(file_location)

        # Run orchestration
        recommendations = orchestrator.run(resume_text)

        return {
            "resume_text": resume_text[:500],  # show preview only
            "recommendations": recommendations
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------------
# Analyze Raw Profile Text
# -------------------------
@app.post("/analyze-profile/")
async def analyze_profile(profile_text: str):
    try:
        recommendations = orchestrator.run(profile_text)
        return {"recommendations": recommendations}
    except Exception as e:
        return {"error": str(e)}
