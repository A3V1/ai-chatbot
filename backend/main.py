import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from cbot import ChatBot

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for request
class MessageRequest(BaseModel):
    message: str

# Pydantic model for response
class MessageResponse(BaseModel):
    response: str

# Initialize chatbot instance (singleton)
bot = ChatBot()

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}

@app.post("/chat", response_model=MessageResponse)
def chat_endpoint(request: MessageRequest):
    user_message = request.message
    bot_response = bot.ask(user_message)
    return {"response": bot_response}