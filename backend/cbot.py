# =============================
# cbot.py - AI Chatbot Logic
# =============================
# This module implements the core logic for the insurance chatbot, including:
# - ChatBot: main class for handling user queries, context, and LLM interaction
# - UserInfoManager: manages user profile data
# - PolicyManager: policy name extraction
# - UserInputProcessor: user intent detection
# - PromptTemplate: prompt for LLM
# - PaymentResponse: structure for payment actions
#
# The chatbot loads user profile and context, uses semantic search for relevant info,
# and generates responses using an LLM. It persists conversation state and history.

import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from data_processing.user_context import (
    get_user_context, load_user_data, get_user_info, save_user_data,
    update_user_context, set_selected_plan, get_selected_plan
)
from jinja2 import Template
from data_processing.mysql_connector import get_mysql_connection, get_policy_brochure_url

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Enum for conversation states in the chat flow."""
    START = 'start'
    RECOMMENDATION_GIVEN = 'recommendation_given'
    SHOWING_DETAILS = 'showing_details'
    DISCUSSING_POLICY = 'discussing_policy'
    AWAITING_APPLICATION_CONFIRMATION = 'awaiting_application_confirmation'
    AWAITING_PAYMENT_CONFIRMATION = 'awaiting_payment_confirmation'
    PAYMENT_INITIATED = 'payment_initiated'


@dataclass
class PaymentResponse:
    """Data class for payment response structure sent to frontend."""
    action: str
    plan: str
    amount: float
    message: str


class ChatBotConfig:
    """Configuration for ChatBot: API keys, model names, and vector DB settings."""
    def __init__(self):
        load_dotenv()
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "aped-4627-b74a")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = "google/gemini-2.5-flash-lite-preview-06-17"  # LLM used for response generation
        self.openai_api_base = "https://openrouter.ai/api/v1"  # OpenRouter API endpoint
        self.temperature = 0.8  # LLM creativity
        self.embedding_model = "all-MiniLM-L6-v2"  # For semantic search
        self.pinecone_index = "insurance-chatbot"  # Pinecone index name
        self.similarity_search_k = 6 # Number of docs to retrieve
        os.environ["PINECONE_ENVIRONMENT"] = self.pinecone_env


class PolicyManager:
    POLICIES = [
        "SmartGrowth ULIP", "InvestSmart ULIP Plan", "MediCare Secure Plan",
        "CarePlus Comprehensive", "WellShield Elite", "HealthGuard Basic",
        "FamilyCare Premium", "SecureTerm Plus", "LifeShield Essential",
        "AutoProtect Comprehensive", "FutureWealth Builder", "HomeSecure Essentials",
        "TotalHome Protection Plan", "AcciGuard Secure", "LifeShield Critical Care"
    ]
    
    @classmethod
    def extract_policy_from_text(cls, text: str) -> Optional[str]:
        """Return the policy name if found in the given text."""
        text_lower = text.lower()
        for policy in cls.POLICIES:
            if policy.lower() in text_lower:
                return policy
        return None


class UserInputProcessor:
    """Processes and categorizes user input for intent detection (affirmative, negative, etc.)."""
    
    YES_VARIANTS = {
        'yes', 'y', 'proceed', 'ok', 'okay', 'sure', 'let\'s go',
        'go ahead', 'continue', 'confirm', 'pay', 'payment', 'ready',
        'show details', 'see details', 'yes please', 'yes, proceed',
        'yes, go ahead', 'yes, pay', 'pay now', 'make payment', 'secure this',
        'let\'s do it', 'i agree', 'i want this', 'book now', 'buy now', 'purchase',
        'yes, i want this', 'yes, i want to pay', 'yes, i want to proceed',
    }
    
    NO_VARIANTS = {'no', 'n'}
    
    APPLICATION_KEYWORDS = {
        'apply', 'i want to apply', 'proceed', 'proceed to application'
    }
    
    PAYMENT_KEYWORDS = {
        'i want to pay', 'payment', 'pay', 'proceed to payment',
        'do payment'
    }
    
    BROCHURE_KEYWORDS = {'brochure', 'details'}
    
    QUESTION_KEYWORDS = {
        'claim', 'exclusion', 'benefit', 'hospital', 'coverage',
        'premium', 'what', 'how', 'when', 'where', 'why'
    }
    
    @classmethod
    def normalize_input(cls, user_input: str) -> str:
        """Lowercase and strip user input for comparison."""
        return user_input.strip().lower()
    
    @classmethod
    def is_affirmative(cls, user_input: str) -> bool:
        """Return True if input is an affirmative response."""
        return cls.normalize_input(user_input) in cls.YES_VARIANTS
    
    @classmethod
    def is_negative(cls, user_input: str) -> bool:
        """Return True if input is a negative response."""
        return cls.normalize_input(user_input) in cls.NO_VARIANTS
    
    @classmethod
    def contains_keywords(cls, user_input: str, keywords: set) -> bool:
        """Return True if any keyword is present in the input."""
        normalized_input = cls.normalize_input(user_input)
        return any(keyword in normalized_input for keyword in keywords)


class PromptTemplate:
    
    RAW_TEMPLATE = """
You are an expert insurance advisor guiding users through: Recommendation -> Application -> Payment.

PRIMARY GOAL: Lead conversations toward successful policy purchase and payment.

CONTEXT: {{ retrieved_context }}
CONVERSATION STATE: {{ conversation_state }}

USER PROFILE:
- Phone: {{ phone_number }}
- Policy Interest: {{ interested_policy_type or 'Not specified' }}
- Age: {{ age }} | Budget: {{ premium_budget }}
- Coverage: {{ desired_coverage }} | Dependents: {{ has_dependents }}
- Experience: {{ insurance_experience_level }}

CHAT HISTORY: {{ chat_history }}
CURRENT QUESTION: {{ question }}

---

RESPONSE STRATEGY (Keep under 40 words):

IF STATE = 'start':
- Use user profile to recommend a policy based on their informations and requirements.
- Format: "PolicyName covers ₹X, ₹Y/year. [Key benefit]. Would you like to apply?"
- if requirements matches a policy recommend a policy if not recommend nearest policy which matches the requirements.

IF STATE = 'recommendation_given':
- give details about the recommended policy.
- If asked about brochure, provide URL and ask if they want to apply.
- Don't repeat recommendations
- Answer questions about the recommended policy
- If no policy matches the requirement: "No policy matches your needs."
-when recommending a policy, always include:
  - Coverage amount and premium
  - Key benefits 

STEP 2 - BUILD INTEREST:
- If asked about brochure: "Here's the brochure: {{ policy_brochure_url }}. This plan suits your profile perfectly. Want to apply?"
- If asked about claims: "Claims via {{ claim_process }}, 24hr processing. Ready to secure your family's future?"
- If asked about exclusions: "Exclusions: {{ exclusions }}. Still excellent coverage for your needs. Shall we proceed?"

STEP 3 - CLOSE FOR APPLICATION:
- Always end responses with application prompts:
  - "Ready to apply and secure this rate?"
  - "Shall we start your application?"

STEP 4 - PAYMENT TRANSITION:
- When user shows interest: "Perfect! I'll start your application. You'll get confirmation shortly. Ready for payment?"

ALWAYS END WITH ACTION:
Every response must include one of these:
- "Want to apply?"
- "Ready to proceed?"
- "Shall we secure this for you?"
- "Ready for payment?"

if user asks about payment:
- "Redirecting to payment page for "policy name' ₹{{ amount }}. Ready to secure your policy?"

---

### ✅ Example Conversation

👤 USER PROFILE:
- Phone: 9876543210  
- Age: 34 | Budget: ₹15,000/year  
- Coverage: ₹1 Cr | Dependents: Yes  
- Interested Policy Type: Term Life  
- Experience: Beginner  

---

🟢 STATE: start  
USER: "I want term insurance for 1 crore. Budget around 15k."  
BOT: "X covers ₹1 Cr, ₹14,800/year. 100% claim settlement, includes terminal illness cover. Would you like to apply?"  
fetch policy details from DB pinecone index
➡️ STATE → recommendation_given

---

🟡 STATE: recommendation_given  
USER: "Can you share the brochure?"  
BOT: "Here's the brochure: https://insure.com/brochures/x.pdf. This plan suits your profile perfectly. Want to apply?"
        fetch policy_brochure_url from DB
---

🟠 STATE: showing_details  
USER: "Yes, show me the benefits."  
BOT:  
"✓ ₹1 Cr coverage  
✓ Terminal illness cover  
✓ 24hr claim settlement  
✓ ₹14,800/year  

Shall we start your application?"  
➡️ STATE → showing_details


🔵 STATE: awaiting_application_confirmation  
USER: "Okay, proceed."  
BOT: "Perfect! Starting your application now. You'll get confirmation shortly. Ready to make payment?"  
➡️ STATE → awaiting_payment_confirmation



🔴 STATE: awaiting_payment_confirmation  
USER: "Yes, go ahead."  
BOT:

  "action": "redirect_to_payment",
  "plan": "x",
  "amount": 14800,
  "message": "Redirecting to payment page..."
  
  you have vehicle insurance with to recommend too.


"""


class ChatBot:
    """
    Main chatbot class. Handles:
    - Loading user profile and context from DB
    - Managing conversation state and chat history
    - Using Pinecone for semantic search
    - Generating responses with LLM
    - Persisting context after each message
    """
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.config = ChatBotConfig()
        self._initialize_components()
        self._load_user_data()        

    def _initialize_components(self) -> None:
        """Initialize LLM, vector store, retriever, and memory buffer."""
        try:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                openai_api_key=self.config.openai_api_key,
                openai_api_base=self.config.openai_api_base,
                temperature=self.config.temperature
            )
            
            embedding = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            
            self.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=self.config.pinecone_index,
                embedding=embedding
            )
            
            self.retriever = self.vectorstore.as_retriever()
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            logger.info("ChatBot components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChatBot components: {e}")
            raise
    
    def _load_user_data(self) -> None:
        """Load user context and chat history from DB, and populate memory."""
        try:
            context, chat_history = load_user_data(self.phone_number)
            self.user_context = context or {}
            
            # Load chat history into memory
            chat_history = chat_history or []
            for msg in chat_history:
                if msg.get("type") == "human":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg.get("type") == "ai":
                    self.memory.chat_memory.add_ai_message(msg["content"])
            
            self.user_info = get_user_info(self.phone_number) or {}
            logger.info(f"User data loaded for phone: {self.phone_number}")
            
        except Exception as e:
            logger.error(f"Error loading user data: {e}")
            self.user_context = {}
            self.user_info = {}
    
    def get_conversation_state(self) -> ConversationState:
        """Fetch the current conversation state from DB (default: START)."""
        user_context = get_user_context(self.phone_number) or {}
        state_str = user_context.get('conversation_state', ConversationState.START.value)
        
        try:
            return ConversationState(state_str)
        except ValueError:
            logger.warning(f"Invalid conversation state: {state_str}. Defaulting to START")
            return ConversationState.START
    
    def set_conversation_state(self, state: ConversationState) -> None:
        """Set and persist the conversation state, and update in-memory context."""
        try:
            update_user_context(self.phone_number, {'conversation_state': state.value})
            self.user_context["conversation_state"] = state.value
            logger.debug(f"Conversation state updated to: {state.value}")
            self._save_context()
        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")
    
    def _save_context(self):
        """Persist conversation state, summary, and chat history to DB after each message."""
        # Update context summary with last bot message (or a simple summary)
        last_bot_msg = None
        for msg in reversed(self.memory.chat_memory.messages):
            if msg.type == "ai":
                last_bot_msg = msg.content
                break
        context_summary = last_bot_msg or self.user_context.get("context_summary", "")
        # Save to DB
        save_user_data(
            self.phone_number,
            context_summary,
            [
                {"type": m.type, "content": m.content}
                for m in self.memory.chat_memory.messages
            ]
        )
        # Update in-memory context
        self.user_context["context_summary"] = context_summary
        self.user_context["conversation_state"] = self.get_conversation_state().value

    def _handle_payment_confirmation(self, query: str) -> Union[PaymentResponse, str]:
        """Handle user input when awaiting payment confirmation."""
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.PAYMENT_INITIATED)
            self.memory.chat_memory.add_user_message(query)
            
            plan = get_selected_plan(self.phone_number)
            amount = self.user_info.get('premium_budget', 1000)
            
            response = PaymentResponse(
                action='redirect_to_payment',
                plan=plan,
                amount=float(amount),
                message='Redirecting to payment page...'
            )
            
            self.memory.chat_memory.add_ai_message(response.message)
            return response
            
        elif UserInputProcessor.is_negative(query):
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            self.memory.chat_memory.add_user_message(query)
            response = 'No problem! Feel free to ask any questions about the policy.'
            self.memory.chat_memory.add_ai_message(response)
            return response
        
        return self._generate_llm_response(query)
    
    def _handle_payment_initiated(self, query: str) -> Union[PaymentResponse, str]:
        """Handle payment initiated state."""
        if UserInputProcessor.contains_keywords(query, UserInputProcessor.PAYMENT_KEYWORDS):
            plan = get_selected_plan(self.phone_number)
            amount = self.user_info.get('premium_budget', 1000)
            
            response = PaymentResponse(
                action='redirect_to_payment',
                plan=plan,
                amount=float(amount),
                message='Redirecting to payment page...'
            )
            
            self.memory.chat_memory.add_ai_message(response.message)
            return response
        
        self.memory.chat_memory.add_user_message(query)
        response = "Your payment is being processed. If you have any more questions or need further assistance, let me know!"
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _handle_application_confirmation(self, query: str) -> str:
        """Handle application confirmation state."""
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
            self.memory.chat_memory.add_user_message(query)
            # Set selected plan in user_context when user confirms application
            last_policy = self._extract_last_policy_from_history()
            logger.info(f"[Application Confirmation] Phone: {self.phone_number}, Extracted Policy: {last_policy}")
            if last_policy:
                result = set_selected_plan(self.phone_number, last_policy)
                logger.info(f"set_selected_plan result: {result}")
            response = "Perfect! Starting your application now. You'll get confirmation shortly. Ready to make payment?"
            self.memory.chat_memory.add_ai_message(response)
            return response
            
        elif UserInputProcessor.is_negative(query):
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            self.memory.chat_memory.add_user_message(query)
            response = 'No worries! Let me know if you have other questions about the policy.'
            self.memory.chat_memory.add_ai_message(response)
            return response
        
        return self._generate_llm_response(query)
    
    def _handle_recommendation_given(self, query: str) -> str:
        """Handle user input after a policy recommendation has been given."""
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            last_policy = self._extract_last_policy_from_history()
            logger.info(f"[Recommendation Given] Phone: {self.phone_number}, Extracted Policy: {last_policy}")
            if last_policy:
                result = set_selected_plan(self.phone_number, last_policy)
                logger.info(f"set_selected_plan result: {result}")
                brochure_url = get_policy_brochure_url(last_policy)
                if brochure_url:
                    response = f"Here's the {last_policy} brochure: {brochure_url}\n\nThis plan is perfect for your profile. Ready to apply and lock in this rate?"
                else:
                    response = f"{last_policy} offers:\n✓ ₹10L coverage\n✓ Cashless hospitals\n✓ No waiting period for accidents\n✓ Family floater option\n\nReady to apply?"
            else:
                response = "Details not available. Please ask about other policies or provide more details."
            self.memory.chat_memory.add_ai_message(response)
            return response
        return self._generate_llm_response(query)
    
    def _handle_brochure_request(self, query: str) -> str:
        """Handle user requests for a policy brochure."""
        self.set_conversation_state(ConversationState.SHOWING_DETAILS)
        last_policy = self._extract_last_policy_from_history()
        self.memory.chat_memory.add_user_message(query)
        
        if last_policy:
            brochure_url = get_policy_brochure_url(last_policy)
            if brochure_url:
                response = f"Here's the {last_policy} brochure: {brochure_url}\n\nReady to apply?"
            else:
                response = f"{last_policy} details:\n✓ Comprehensive coverage\n✓ Cashless claims\n✓ Wide network\n\nReady to apply?"
        else:
            response = "I'll get you the policy details. Ready to apply after reviewing?"
        
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _handle_application_request(self, query: str) -> str:
        """Handle user requests to apply for a policy."""
        self.set_conversation_state(ConversationState.AWAITING_APPLICATION_CONFIRMATION)
        self.memory.chat_memory.add_user_message(query)
        response = "Excellent choice! I can start your application right now. Shall we proceed with your details?"
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _handle_payment_request(self, query: str) -> str:
        """Handle user requests to proceed to payment."""
        self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
        self.memory.chat_memory.add_user_message(query)
        response = "Ready to secure your policy!"
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _generate_llm_response(self, query: str) -> str:
        """
        Generate a response using the LLM.
        - Performs semantic search for relevant docs
        - Builds a prompt with user profile, context, and chat history
        - Invokes the LLM and returns its response
        """
        try:
            retrieved_docs = self.vectorstore.similarity_search(
                query, k=self.config.similarity_search_k
            )
            retrieved_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])
            
            chat_history_str = ""
            for msg in self.memory.chat_memory.messages:
                if msg.type == "human":
                    chat_history_str += f"User: {msg.content}\n"
                elif msg.type == "ai":
                    chat_history_str += f"Bot: {msg.content}\n"
            
            merged_context = {
                **self.user_context,
                **self.user_info,
                "retrieved_context": retrieved_context,
                "chat_history": chat_history_str,
                "question": query,
                "conversation_state": self.get_conversation_state().value,
            }
            
            full_prompt = Template(PromptTemplate.RAW_TEMPLATE).render(merged_context)
            response = self.llm.invoke(full_prompt)
            
            self.memory.chat_memory.add_user_message(query)
            self.memory.chat_memory.add_ai_message(response.content)
            
            # --- Set selected_plan as soon as a valid policy is mentioned in the AI response ---
            last_policy = self._extract_last_policy_from_history()
            if last_policy:
                logger.info(f"[LLM Response] Phone: {self.phone_number}, Extracted Policy: {last_policy}")
                result = set_selected_plan(self.phone_number, last_policy)
                logger.info(f"set_selected_plan result: {result}")
            
            return response.content
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def _extract_last_policy_from_history(self) -> Optional[str]:
        """Extract the latest policy name from chat history that matches POLICIES, cleaning formatting."""
        import re
        policies = PolicyManager.POLICIES
        for msg in reversed(self.memory.chat_memory.messages):
            # Remove markdown/bold/asterisks and extra spaces
            content_clean = re.sub(r'[\*`_]', '', msg.content).strip().lower()
            for policy in policies:
                if policy.lower() in content_clean:
                    logger.debug(f"Matched policy '{policy}' in message: {msg.content}")
                    return policy
        # Fallback: check user_context
        policy = self.user_context.get('selected_plan')
        if policy:
            logger.debug(f"Fallback to user_context selected_plan: {policy}")
            return policy
        logger.debug("No policy found in chat history or user_context.")
        return None
    
    def ask(self, query: str) -> Union[str, PaymentResponse]:
        """
        Main entry point for user queries.
        - Routes input based on conversation state and intent
        - Persists context after every message
        - Returns either a string (bot reply) or PaymentResponse
        """
        if not query.strip():
            return "Please provide a valid question or request."
        
        state = self.get_conversation_state()
        logger.debug(f"Processing query in state: {state.value}")
        
        try:
            # State-based routing
            if state == ConversationState.AWAITING_PAYMENT_CONFIRMATION:
                result = self._handle_payment_confirmation(query)
            elif state == ConversationState.PAYMENT_INITIATED:
                result = self._handle_payment_initiated(query)
            elif state == ConversationState.AWAITING_APPLICATION_CONFIRMATION:
                result = self._handle_application_confirmation(query)
            elif state == ConversationState.RECOMMENDATION_GIVEN:
                result = self._handle_recommendation_given(query)
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.APPLICATION_KEYWORDS):
                result = self._handle_application_request(query)
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.PAYMENT_KEYWORDS):
                result = self._handle_payment_request(query)
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.BROCHURE_KEYWORDS):
                result = self._handle_brochure_request(query)
            elif state == ConversationState.START:
                self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
                result = self._generate_llm_response(query)
            elif (state in [ConversationState.RECOMMENDATION_GIVEN, ConversationState.SHOWING_DETAILS] and
                  UserInputProcessor.contains_keywords(query, UserInputProcessor.QUESTION_KEYWORDS)):
                result = self._generate_llm_response(query)
            else:
                response = "I'm here to help with your insurance needs. Ready to apply or have specific questions?"
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
                result = response
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            result = "I'm here to help you secure your insurance. You can ask about policy details, say 'yes' to proceed, or type 'pay' to move to payment."
        # Persist context after every message
        self._save_context()
        return result


class UserInfoManager:
    """
    Manages user profile data in the user_info table:
    - Checks for missing fields
    - Updates user info
    - Creates new user entries
    """
    REQUIRED_FIELDS = [
        ("interested_policy_type", "What type of insurance are you interested in?"),
        ("age", "What's your age?"),
        ("desired_coverage", "What is the coverage amount you are looking for?"),
        ("premium_budget", "What is your budget for annual premium?"),
        ("premium_payment_mode", "How would you prefer to pay premiums (Annually, Semi-Annually, Monthly)?"),
        ("preferred_add_ons", "Are there any benefits you want (like accidental death cover, waiver of premium, etc.)?"),
        ("has_dependents", "Do you have any dependents (e.g., spouse, children, parents)? (yes/no)"),
        ("policy_duration_years", "How long would you like the policy coverage to last (in years)?"),
        ("insurance_experience_level", "Have you purchased insurance before?"),
    ]
    
    @classmethod
    def get_missing_fields(cls, user_info: Dict[str, Any]) -> List[Tuple[str, str]]:
        """Get missing user information fields"""
        return [
            (field, question) for field, question in cls.REQUIRED_FIELDS
            if user_info.get(field) in [None, "", "null"]
        ]
    
    @classmethod
    def update_user_info(cls, phone_number: str, field: str, value: str) -> bool:
        """Update user information in database"""
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE user_info SET {field} = %s WHERE phone_number = %s",
                (value, phone_number)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Updated {field} for user {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error updating user info: {e}")
            return False
    
    @classmethod
    def create_user(cls, phone_number: str) -> bool:
        """Create a new user entry"""
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO user_info (phone_number) VALUES (%s)",
                (phone_number,)
            )
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Created new user: {phone_number}")
            return True
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False


# For FastAPI, expose the correct helpers from the class

def get_missing_user_info_fields(user_info: dict) -> list:
    """
    Returns a list of (field, question) tuples for missing user info fields.
    """
    from cbot import UserInfoManager
    return UserInfoManager.get_missing_fields(user_info)


def update_user_info(phone_number: str, field: str, value: str) -> bool:
    from cbot import UserInfoManager
    return UserInfoManager.update_user_info(phone_number, field, value)


def main():
    """Main function to run the chatbot"""
    print("=== Insurance Chatbot ===")
    
    phone_number = input("Enter your phone number: ").strip()
    if not phone_number:
        print("Phone number is required to start the chat.")
        return
    
    # Check if user exists, create if not
    user_info = get_user_info(phone_number)
    if not user_info:
        print("Creating new user entry...")
        if not UserInfoManager.create_user(phone_number):
            print("Failed to create user. Please try again.")
            return
        user_info = get_user_info(phone_number)
    
    # Collect missing user information
    while True:
        missing_fields = UserInfoManager.get_missing_fields(user_info)
        if not missing_fields:
            break
        
        for field, question in missing_fields:
            value = input(question + " ").strip()
            if value.lower() in ["exit", "quit"]:
                return
            
            if not UserInfoManager.update_user_info(phone_number, field, value):
                print(f"Failed to update {field}. Please try again.")
            else:             continue
        user_info = get_user_info(phone_number)
    
    # Start chat
    print("\n=== Chat Started ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    
    try:
        bot = ChatBot(phone_number)
        
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("Thank you for using our insurance chatbot!")
                break
            
            if not user_input:
                continue
            
            response = bot.ask(user_input)
            
            if isinstance(response, PaymentResponse):
                print(f"Bot: {response.message}")
                print(f"Action: {response.action}")
                print(f"Plan: {response.plan}")
                print(f"Amount: ₹{response.amount}")
            else:
                print(f"Bot: {response}")
    
    except KeyboardInterrupt:
        print("\n\nChat interrupted by user.")
    except Exception as e:
        logger.error(f"Error in main chat loop: {e}")
        print("An error occurred. Please restart the application.")


if __name__ == "__main__":
    main()