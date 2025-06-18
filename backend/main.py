import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from cbot import ChatBot, get_missing_user_info_fields, update_user_info, get_user_info
from data_processing.mysql_connector import get_mysql_connection
from payments import router as payments_router

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
    missing_fields: Optional[List[Dict[str, str]]] = None
    action: Optional[str] = None
    plan: Optional[str] = None
    amount: Optional[int] = None
    
class ChatRequest(BaseModel):
    message: str
    phone_number: str

class UserInfoUpdate(BaseModel):
    phone_number: str
    field: str
    value: str

@app.get("/")
def root():
    return {"message": "Chatbot API is running"}

@app.post("/initialize_user/{phone_number}")
async def initialize_user(phone_number: str):
    user_info = get_user_info(phone_number)
    if not user_info:
        # Create new user
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_info (phone_number) VALUES (%s)", (phone_number,))
        conn.commit()
        cursor.close()
        conn.close()
        user_info = get_user_info(phone_number)
    
    # Get missing fields
    missing_fields = get_missing_user_info_fields(user_info)
    return {
        "user_exists": bool(user_info),
        "missing_fields": [{"field": field, "question": question} for field, question in missing_fields]
    }

@app.post("/update_user_info")
async def update_user_info_endpoint(update_request: UserInfoUpdate):
    # Simply update the field without any validation
    update_user_info(
        update_request.phone_number,
        update_request.field,
        update_request.value
    )
    
    # Get updated user info and check remaining missing fields
    user_info = get_user_info(update_request.phone_number)
    missing_fields = get_missing_user_info_fields(user_info)
    
    return {
        "success": True,
        "missing_fields": [{"field": field, "question": question} for field, question in missing_fields]
    }

@app.post("/chat", response_model=MessageResponse)
async def chat_endpoint(request: ChatRequest):
    user_message = request.message
    phone_number = request.phone_number
    
    # Handle exit message
    if user_message.lower() == 'exit':
        return MessageResponse(
            response="Goodbye! Chat session ended.",
            missing_fields=None
        )
    
    # Check for missing fields but don't validate their content
    user_info = get_user_info(phone_number)
    missing_fields = get_missing_user_info_fields(user_info)
    
    if missing_fields:
        return MessageResponse(
            response="Please provide the required information to continue.",
            missing_fields=[{"field": field, "question": question} for field, question in missing_fields]
        )
    
    bot = ChatBot(phone_number)
    bot_response = bot.ask(user_message)
    # If bot_response is a dict (for payment redirect), unpack fields
    if isinstance(bot_response, dict):
        return MessageResponse(
            response=bot_response.get('message', ''),
            action=bot_response.get('action'),
            plan=bot_response.get('plan'),
            amount=bot_response.get('amount'),
            missing_fields=None
        )
    return MessageResponse(response=bot_response, missing_fields=None)

@app.get("/plan_details/{phone_number}")
def get_plan_details(phone_number: str):
    from data_processing.user_context import get_selected_plan
    plan = get_selected_plan(phone_number)
    # TODO: Replace with real plan lookup from DB or config
    # For now, use user_info budget as amount
    user_info = get_user_info(phone_number)
    amount = user_info.get('premium_budget', 1000)
    return {"plan": plan, "amount": amount}

app.include_router(payments_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
