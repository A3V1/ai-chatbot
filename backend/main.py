import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from cbot import ChatBot, get_missing_user_info_fields, update_user_info, get_user_info, get_policy_specific_questions, save_policy_specific_qa, get_interested_policy_type, save_interested_policy_type
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

class PolicySpecificAnswersRequest(BaseModel):
    phone_number: str
    answers: dict

class InterestedPolicyTypeRequest(BaseModel):
    phone_number: str
    interested_policy_type: str

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
    Prevents updating interested_policy_type (must use /save_interested_policy_type).
    Rejects fields not present in user_info schema.
    """
    if update_request.field == "interested_policy_type":
        return {
            "success": False,
            "error": "interested_policy_type must be set using /save_interested_policy_type, not /update_user_info."
        }
    # Only allow fields that are in the user_info table
    allowed_fields = set([
        "age",
        "desired_coverage",
        "premium_budget",
        "premium_payment_mode",
        "preferred_add_ons",
        "has_dependents",
        "policy_duration_years",
        "insurance_experience_level"
    ])
    if update_request.field not in allowed_fields:
        # Log and return error, do NOT call update_user_info at all
        import logging
        logging.error(f"Attempted to update disallowed field: {update_request.field}")
        return {
            "success": False,
            "error": f"Field '{update_request.field}' is not allowed to be updated in user_info. Use the appropriate endpoint for policy-specific answers."
        }
    # Only update allowed fields
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
        # Save summary and chat history on exit
        from data_processing.user_context import save_user_data, load_user_data
        bot = ChatBot(phone_number)
        chat_history_to_save = []
        for msg in bot.memory.chat_memory.messages:
            chat_history_to_save.append({"type": msg.type, "content": msg.content})
        current_user_context = bot.user_context.get('context_summary', '')
        # Load existing chat history and append new messages
        _, existing_history = load_user_data(phone_number)
        combined_history = existing_history.copy()
        for msg in chat_history_to_save:
            if msg not in combined_history:
                combined_history.append(msg)
        save_user_data(phone_number, current_user_context, combined_history)
        return MessageResponse(
            response="Goodbye! Chat session ended.",
            missing_fields=None
        )
    
    # Check for missing fields but don't validate their content
    user_info = get_user_info(phone_number)
    missing_fields = get_missing_user_info_fields(user_info)

    # Check for interested_policy_type using chat_history (not user_info)
    interested_policy_type = get_interested_policy_type(phone_number)
    policy_type_map = {
        "health": "Health",
        "term": "Term Life",
        "term life": "Term Life",
        "investment": "Investment",
        "vehicle": "Vehicle",
        "home": "Home"
    }
    user_policy_type = user_message.strip().lower()
    mapped_policy_type = policy_type_map.get(user_policy_type)
    if mapped_policy_type:
        save_interested_policy_type(phone_number, mapped_policy_type)
        # Reload interested_policy_type and chat_history after saving
        interested_policy_type = get_interested_policy_type(phone_number)
        from data_processing.user_context import load_user_data
        context, chat_history = load_user_data(phone_number)
    elif not interested_policy_type: # Only ask if no policy type is set yet
        return MessageResponse(
            response="What type of insurance are you looking for? Health, Term Life, Investment, Vehicle, or Home?",
            missing_fields=[{"field": "interested_policy_type", "question": "What type of insurance are you looking for? Health, Term Life, Investment, Vehicle, or Home?"}]
        )

    # Check for policy-specific answers in chat_history only
    from cbot import get_policy_specific_questions
    from data_processing.user_context import load_user_data
    context, chat_history = load_user_data(phone_number)
    policy_questions = get_policy_specific_questions(interested_policy_type)
    # Gather all answered fields from chat_history
    answered_fields = set()
    for msg in (chat_history or []):
        if msg.get("type") == "policy_specific_answers":
            qa = msg.get("qa")
            if qa and isinstance(qa, list):
                for item in qa:
                    if item.get("field") and item.get("answer") not in [None, ""]:
                        answered_fields.add(item["field"])
            # Also support old format: {field: value}
            content = msg.get("content")
            if content and isinstance(content, dict):
                for k, v in content.items():
                    if v not in [None, ""]:
                        answered_fields.add(k)
    # Find missing policy-specific fields
    missing_policy_fields = [q for q in policy_questions if q["field"] not in answered_fields]
    if missing_policy_fields:
        return MessageResponse(
            response="Please answer the following questions to help us recommend the best policy.",
            missing_fields=missing_policy_fields
        )
    
    # If all user info and policy-specific questions are answered, trigger a recommendation
    from data_processing.user_context import load_user_data, update_user_context
    user_context_data, _ = load_user_data(phone_number)
    info_taken = user_context_data.get('info_taken', False) if user_context_data else False
    if not missing_fields and not missing_policy_fields and interested_policy_type and not info_taken:
        # Override user_message to force a recommendation
        user_message_for_bot = "recommend a policy for me"
        # Set info_taken to True in user_context
        update_user_context(phone_number, {'info_taken': True})
    else:
        user_message_for_bot = user_message
    
    # Pass the loaded context and chat_history to the ChatBot constructor
    bot = ChatBot(phone_number, initial_context=context, initial_chat_history=chat_history)
    bot_response = bot.ask(user_message_for_bot)
    
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

@app.get("/policy_specific_questions")
def api_policy_specific_questions(interested_policy_type: str):
    """
    API endpoint to fetch policy-type-specific questions for the UI.
    """
    return {"questions": get_policy_specific_questions(interested_policy_type)}

@app.post("/save_policy_specific_answers")
def api_save_policy_specific_answers(request: PolicySpecificAnswersRequest):
    
    interested_policy_type = get_interested_policy_type(request.phone_number)
    save_policy_specific_qa(request.phone_number, interested_policy_type, request.answers)
    # After saving, check for next missing policy-specific field
    from cbot import get_policy_specific_questions
    from data_processing.user_context import load_user_data
    context, chat_history = load_user_data(request.phone_number)
    policy_questions = get_policy_specific_questions(interested_policy_type)
    answered_fields = set()
    for msg in (chat_history or []):
        if msg.get("type") == "policy_specific_answers":
            qa = msg.get("qa")
            if qa and isinstance(qa, list):
                for item in qa:
                    if item.get("field"):
                        answered_fields.add(item["field"])
            content = msg.get("content")
            if content and isinstance(content, dict):
                for k in content.keys():
                    answered_fields.add(k)
    missing_policy_fields = [q for q in policy_questions if q["field"] not in answered_fields]
    if missing_policy_fields:
        return {"success": True, "next_question": missing_policy_fields[0]}
    return {"success": True}

@app.get("/get_interested_policy_type")
def api_get_interested_policy_type(phone_number: str):
    """
    API endpoint to get the interested_policy_type for a user.
    """
    return {"interested_policy_type": get_interested_policy_type(phone_number)}

@app.post("/save_interested_policy_type")
def api_save_interested_policy_type(request: InterestedPolicyTypeRequest):
    """
    API endpoint to save/update the interested_policy_type for a user.
    """
    save_interested_policy_type(request.phone_number, request.interested_policy_type)
    return {"success": True}

app.include_router(payments_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
