from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mentor import tools
import os
import json
import uvicorn
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Mentor API", description="AI Career Mentor with Tools")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Request models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str

class ToolInfo(BaseModel):
    name: str
    description: str

def call_groq_api(messages, tools=None, tool_choice=None):
    """Call Groq API with the given messages and optional tools"""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": messages,
        "model": "llama-3.1-8b-instant",  # You can change to "llama3-8b-8192" for faster responses
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 0.9,
        "stream": False
    }
    
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Groq API error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Mentor API is running successfully",
        "version": "1.0.0",
        "status": "healthy",
        "description": "AI Career Mentor powered by Groq",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs",
            "openapi": "/openapi.json"
        },
        "model": "llama3-70b-8192",
        "tools": ["web_search", "job_search", "wellness_guide", "calendar"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test Groq API connection with a simple call
        test_messages = [{"role": "user", "content": "Hello"}]
        response = call_groq_api(test_messages)
        
        if response.get("choices"):
            return HealthResponse(
                status="healthy", 
                message="Mentor API is healthy - Groq AI initialized and ready"
            )
        else:
            return HealthResponse(
                status="degraded", 
                message="Mentor API running but Groq API response format unexpected"
            )
    except Exception as e:
        return HealthResponse(
            status="unhealthy", 
            message=f"Health check failed: {str(e)}"
        )

def should_use_tools(message: str) -> bool:
    """Determine if the message requires tool usage"""
    tool_keywords = [
        'search', 'find', 'lookup', 'job', 'career', 'position', 'role',
        'wellness', 'stress', 'anxiety', 'mental health', 'calm',
        'calendar', 'schedule', 'meeting', 'appointment', 'invite',
        'web', 'internet', 'online', 'latest', 'current', 'trend'
    ]
    
    message_lower = message.lower()
    
    # Check if message contains tool-related keywords
    for keyword in tool_keywords:
        if keyword in message_lower:
            return True
    
    # Simple questions that don't need tools
    simple_questions = [
        'how are you', 'what is', 'who is', 'when is', 'where is',
        'why', 'explain', 'tell me about', 'define', 'describe'
    ]
    
    for question in simple_questions:
        if message_lower.startswith(question):
            return False
    
    return False

def get_fallback_response(message: str) -> str:
    """Get response using Groq API for general questions"""
    system_prompt = """You are a helpful AI career mentor assistant. Provide helpful, 
    informative, and supportive responses to general questions. Be professional yet 
    friendly in your tone. If you don't know something, admit it honestly."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": message}
    ]
    
    try:
        response = call_groq_api(messages)
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your question right now. Error: {str(e)}"

def convert_tools_to_openai_format():
    """Convert LangChain tools to OpenAI tool format for Groq API"""
    openai_tools = []
    for tool in tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        # Add parameters if available (simplified - you may need to adjust based on your tool schemas)
        if hasattr(tool, 'args_schema'):
            # You'll need to implement proper schema conversion based on your tool definitions
            pass
            
        openai_tools.append(openai_tool)
    
    return openai_tools

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
        
    try:
        message = request.message
        
        if not message:
            raise HTTPException(status_code=400, detail="Message is required")
        
        # Check if we should use tools or fallback
        if should_use_tools(message):
            try:
                # Convert tools to OpenAI format
                openai_tools = convert_tools_to_openai_format()
                
                # First call to see if tools are needed
                messages = [{"role": "user", "content": message}]
                response = call_groq_api(messages, tools=openai_tools, tool_choice="auto")
                
                message_response = response["choices"][0]["message"]
                
                # Check if tool calls are needed
                if message_response.get("tool_calls"):
                    tool_results = []
                    for tool_call in message_response["tool_calls"]:
                        tool_name = tool_call["function"]["name"]
                        tool_args = json.loads(tool_call["function"]["arguments"])
                        
                        # Find and execute the tool
                        for tool in tools:
                            if tool.name == tool_name:
                                result = tool.invoke(tool_args)
                                tool_results.append(f"{tool_name} result: {result}")
                                break
                    
                    # If we have tool results, send them to the LLM for a final response
                    if tool_results:
                        follow_up_message = f"Original query: {message}\n\nTool results:\n" + "\n".join(tool_results)
                        follow_up_messages = [
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": "", "tool_calls": message_response["tool_calls"]},
                            {"role": "tool", "content": "\n".join(tool_results)}
                        ]
                        final_response = call_groq_api(follow_up_messages)
                        return ChatResponse(
                            response=final_response["choices"][0]["message"]["content"], 
                            status="success"
                        )
                
                # Return direct response if no tools were called
                return ChatResponse(response=message_response["content"], status="success")
                
            except Exception as tool_error:
                # If tool usage fails, fall back to regular response
                print(f"Tool usage failed, falling back: {tool_error}")
                response = get_fallback_response(message)
                return ChatResponse(response=response, status="success")
        else:
            # Use direct response for general questions
            response = get_fallback_response(message)
            return ChatResponse(response=response, status="success")
        
    except Exception as e:
        # If anything fails, use the fallback
        try:
            fallback_response = get_fallback_response(message)
            return ChatResponse(response=fallback_response, status="success")
        except:
            return ChatResponse(
                response="I apologize, but I'm experiencing technical difficulties. Please try again later.",
                status="error"
            )

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)