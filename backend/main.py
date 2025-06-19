"""
main.py - FastAPI API Layer for AI Insurance Chatbot
---------------------------------------------------
- Exposes REST endpoints for chatbot interaction, user info, and payments
- Handles user initialization, info update, chat, and plan details
- Delegates business logic to cbot.py and user_context.py
- Ensures context-aware, persistent, and personalized conversations
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from cbot import ChatBot, get_missing_user_info_fields, update_user_info, get_user_info
from data_processing.mysql_connector import get_mysql_connection
from payments import router as payments_router

# Initialize FastAPI app
app = FastAPI()

# Allow frontend (React, etc.) to access API
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Pydantic Models for Request/Response
# =============================
class MessageRequest(BaseModel):
    """Request model for a simple message (not used in main chat)."""
    message: str

class MessageResponse(BaseModel):
    """
    Response model for chat endpoint.
    - response: main bot reply
    - missing_fields: list of required user info fields (if any)
    - action/plan/amount: for payment actions
    """
    response: str
    missing_fields: Optional[List[Dict[str, str]]] = None
    action: Optional[str] = None
    plan: Optional[str] = None
    amount: Optional[int] = None
    
class ChatRequest(BaseModel):
    """Request model for chat endpoint (message + phone number)."""
    message: str
    phone_number: str

class UserInfoUpdate(BaseModel):
    """Request model for updating a single user info field."""
    phone_number: str
    field: str
    value: str

# =============================
# API Endpoints
# =============================
@app.get("/")
def root():
    """Health check endpoint."""
    return {"message": "Chatbot API is running"}

@app.post("/initialize_user/{phone_number}")
async def initialize_user(phone_number: str):
    """
    Initialize user session:
    - Checks if user exists in user_info
    - Creates user if not present
    - Returns missing profile fields for onboarding
    """
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
    # Get missing fields for onboarding
    missing_fields = get_missing_user_info_fields(user_info)
    return {
        "user_exists": bool(user_info),
        "missing_fields": [{"field": field, "question": question} for field, question in missing_fields]
    }

@app.post("/update_user_info")
async def update_user_info_endpoint(update_request: UserInfoUpdate):
    """
    Update a single user info field in user_info table.
    Returns updated list of missing fields.
    """
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
    """
    Main chat endpoint:
    - Checks for missing user info fields
    - If all info present, instantiates ChatBot and processes message
    - On 'exit', saves summary and chat history
    - Returns bot response and any required actions (e.g., payment)
    """
    user_message = request.message
    phone_number = request.phone_number

    # Handle exit message
    if user_message.lower() == 'exit':
        # Save summary and chat history on exit
        from data_processing.user_context import save_user_data
        bot = ChatBot(phone_number)
        chat_history_to_save = []
        for msg in bot.memory.chat_memory.messages:
            chat_history_to_save.append({"type": msg.type, "content": msg.content})
        current_user_context = bot.user_context.get('context_summary', '')
        save_user_data(phone_number, current_user_context, chat_history_to_save)
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
    # If bot_response is a PaymentResponse, unpack fields
    from cbot import PaymentResponse
    if isinstance(bot_response, PaymentResponse):
        return MessageResponse(
            response=bot_response.message,
            action=bot_response.action,
            plan=bot_response.plan,
            amount=bot_response.amount,
            missing_fields=None
        )
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

# @app.get("/plan_details/{phone_number}")
# def get_plan_details(phone_number: str):
#     """
#     Fetch selected plan and amount for the user (for payment display).
#     """
#     from data_processing.mysql_connector import get_selected_plan
#     from data_processing.mysql_connector import get_policy_premium
#     plan = get_selected_plan(phone_number)
#     amount = None
#     if plan:
#         amount = get_policy_premium(plan)
#     return {"plan": plan, "amount": amount}

@app.get("/api/payment-details")
def api_payment_details(phone_number: str):
    """
    API endpoint to fetch selected plan and premium amount for payment page.
    Adds debug logging for troubleshooting.
    """
    from data_processing.mysql_connector import get_selected_plan
    from data_processing.mysql_connector import get_policy_premium
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(f"Received payment-details request for phone_number: {phone_number}")
    selected_plan = get_selected_plan(phone_number)
    logging.debug(f"Selected plan for {phone_number}: {selected_plan}")
    amount = None
    if selected_plan:
        amount = get_policy_premium(selected_plan)
        logging.debug(f"Premium for plan '{selected_plan}': {amount}")
    else:
        logging.debug(f"No plan found for phone_number: {phone_number}")
    return {"selected_plan": selected_plan, "amount": amount}

# Include payment-related endpoints from payments.py
app.include_router(payments_router)

# =============================
# Run the API (for local dev)
# =============================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
