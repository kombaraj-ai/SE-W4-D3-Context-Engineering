# Before exeuting this code install: "uv add "strands-agents[openai]" python-dotenv"
# strands-agents[openai] → enables built-in OpenAI support
import os 
from dotenv import load_dotenv 
 
# Load environment variables 
load_dotenv() 
 
# Optional safety check 
if not os.getenv("OPENAI_API_KEY"): 
    raise ValueError("OPENAI_API_KEY not found in .env") 
 
from strands import Agent 
from strands.models.openai import OpenAIModel 
 
# Initialize OpenAI model with correct parameter format
model = OpenAIModel( 
    model_id="gpt-4o-mini",  # Use model_id instead of model
    params={
        "temperature": 0.7
    }
) 
 
# Create agent 
agent = Agent( 
    model=model, 
    system_prompt="You are a helpful AI assistant" 
) 
 
# Invoke agent (use __call__ or invoke, not run)
response = agent("Explain Agentic AI in simple terms") 
print(response)