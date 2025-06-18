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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """Enum for conversation states"""
    START = 'start'
    RECOMMENDATION_GIVEN = 'recommendation_given'
    SHOWING_DETAILS = 'showing_details'
    AWAITING_APPLICATION_CONFIRMATION = 'awaiting_application_confirmation'
    AWAITING_PAYMENT_CONFIRMATION = 'awaiting_payment_confirmation'
    PAYMENT_INITIATED = 'payment_initiated'


@dataclass
class PaymentResponse:
    """Data class for payment response"""
    action: str
    plan: str
    amount: float
    message: str


class ChatBotConfig:
    """Configuration class for ChatBot"""
    
    def __init__(self):
        load_dotenv()
        self.pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "aped-4627-b74a")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = "google/gemini-2.5-flash-lite-preview-06-17"
        self.openai_api_base = "https://openrouter.ai/api/v1"
        self.temperature = 0.8
        self.embedding_model = "all-MiniLM-L6-v2"
        self.pinecone_index = "insurance-chatbot"
        self.similarity_search_k = 6
        
        # Set environment variables
        os.environ["PINECONE_ENVIRONMENT"] = self.pinecone_env


class PolicyManager:
    """Manages policy-related operations"""
    
    POLICIES = [
        "SmartGrowth ULIP", "InvestSmart ULIP Plan", "MediCare Secure Plan",
        "CarePlus Comprehensive", "WellShield Elite", "HealthGuard Basic",
        "FamilyCare Premium", "SecureTerm Plus", "LifeShield Essential",
        "AutoProtect Comprehensive", "FutureWealth Builder", "HomeSecure Essentials",
        "TotalHome Protection Plan", "AcciGuard Secure", "LifeShield Critical Care"
    ]
    
    @classmethod
    def extract_policy_from_text(cls, text: str) -> Optional[str]:
        """Extract policy name from text"""
        text_lower = text.lower()
        for policy in cls.POLICIES:
            if policy.lower() in text_lower:
                return policy
        return None


class UserInputProcessor:
    """Processes and categorizes user input"""
    
    YES_VARIANTS = {
        'yes', 'y', 'proceed', 'ok', 'okay', 'sure', 'let\'s go',
        'go ahead', 'continue', 'confirm', 'pay', 'payment', 'ready',
        'show details', 'see details'
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
        """Normalize user input"""
        return user_input.strip().lower()
    
    @classmethod
    def is_affirmative(cls, user_input: str) -> bool:
        """Check if user input is affirmative"""
        return cls.normalize_input(user_input) in cls.YES_VARIANTS
    
    @classmethod
    def is_negative(cls, user_input: str) -> bool:
        """Check if user input is negative"""
        return cls.normalize_input(user_input) in cls.NO_VARIANTS
    
    @classmethod
    def contains_keywords(cls, user_input: str, keywords: set) -> bool:
        """Check if user input contains any of the keywords"""
        normalized_input = cls.normalize_input(user_input)
        return any(keyword in normalized_input for keyword in keywords)


class PromptTemplate:
    """Manages prompt templates"""
    
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
- Give ONE policy recommendation only if any policy matches user requirements from the database otherwise say "No policy matches your needs."
- Use user profile to recommend a policy.
- Format: "PolicyName covers ₹X, ₹Y/year. [Key benefit]. Would you like to apply?"

IF STATE = 'recommendation_given':
- Don't repeat recommendations
- Answer questions about the recommended policy
- Always push toward: "Ready to apply?"
- If no policy matches the requirement: "No policy matches your needs."

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
BOT: "WellShield Elite covers ₹1 Cr, ₹14,800/year. 100% claim settlement, includes terminal illness cover. Would you like to apply?"  
➡️ STATE → recommendation_given

---

🟡 STATE: recommendation_given  
USER: "Can you share the brochure?"  
BOT: "Here's the brochure: https://insure.com/brochures/wellshield-elite.pdf. This plan suits your profile perfectly. Want to apply?"

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
  "plan": "WellShield Elite",
  "amount": 14800,
  "message": "Redirecting to payment page..."


"""


class ChatBot:
    """Main chatbot class with improved error handling and structure"""
    
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.config = ChatBotConfig()
        self._initialize_components()
        self._load_user_data()
    
    def _initialize_components(self) -> None:
        """Initialize LLM, vector store, and other components"""
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
        """Load user context and chat history"""
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
        """Get current conversation state"""
        user_context = get_user_context(self.phone_number) or {}
        state_str = user_context.get('conversation_state', ConversationState.START.value)
        
        try:
            return ConversationState(state_str)
        except ValueError:
            logger.warning(f"Invalid conversation state: {state_str}. Defaulting to START")
            return ConversationState.START
    
    def set_conversation_state(self, state: ConversationState) -> None:
        """Set conversation state"""
        try:
            update_user_context(self.phone_number, {'conversation_state': state.value})
            logger.debug(f"Conversation state updated to: {state.value}")
        except Exception as e:
            logger.error(f"Error updating conversation state: {e}")
    
    def _handle_payment_confirmation(self, query: str) -> Union[PaymentResponse, str]:
        """Handle payment confirmation state"""
        normalized_query = UserInputProcessor.normalize_input(query)
        
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
        """Handle payment initiated state"""
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
        """Handle application confirmation state"""
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
            self.memory.chat_memory.add_user_message(query)
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
        """Handle recommendation given state"""
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            last_policy = self._extract_last_policy_from_history()
            
            if last_policy:
                set_selected_plan(self.phone_number, last_policy)
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
        """Handle brochure requests"""
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
        """Handle application requests"""
        self.set_conversation_state(ConversationState.AWAITING_APPLICATION_CONFIRMATION)
        self.memory.chat_memory.add_user_message(query)
        response = "Excellent choice! I can start your application right now. Shall we proceed with your details?"
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _handle_payment_request(self, query: str) -> str:
        """Handle payment requests"""
        self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
        self.memory.chat_memory.add_user_message(query)
        response = "Ready to secure your policy!"
        self.memory.chat_memory.add_ai_message(response)
        return response
    
    def _generate_llm_response(self, query: str) -> str:
        """Generate response using LLM"""
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
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def _extract_last_policy_from_history(self) -> Optional[str]:
        """Extract the last recommended policy from chat history"""
        for msg in reversed(self.memory.chat_memory.messages):
            if msg.type == "ai":
                policy = PolicyManager.extract_policy_from_text(msg.content)
                if policy:
                    return policy
        return None
    
    def ask(self, query: str) -> Union[str, PaymentResponse]:
        """Main method to process user query"""
        if not query.strip():
            return "Please provide a valid question or request."
        
        state = self.get_conversation_state()
        logger.debug(f"Processing query in state: {state.value}")
        
        try:
            # State-based routing
            if state == ConversationState.AWAITING_PAYMENT_CONFIRMATION:
                return self._handle_payment_confirmation(query)
            
            elif state == ConversationState.PAYMENT_INITIATED:
                return self._handle_payment_initiated(query)
            
            elif state == ConversationState.AWAITING_APPLICATION_CONFIRMATION:
                return self._handle_application_confirmation(query)
            
            elif state == ConversationState.RECOMMENDATION_GIVEN:
                return self._handle_recommendation_given(query)
            
            # Handle specific user intents across states
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.APPLICATION_KEYWORDS):
                return self._handle_application_request(query)
            
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.PAYMENT_KEYWORDS):
                return self._handle_payment_request(query)
            
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.BROCHURE_KEYWORDS):
                return self._handle_brochure_request(query)
            
            # Handle initial state or questions
            elif state == ConversationState.START:
                self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
                return self._generate_llm_response(query)
            
            elif (state in [ConversationState.RECOMMENDATION_GIVEN, ConversationState.SHOWING_DETAILS] and
                  UserInputProcessor.contains_keywords(query, UserInputProcessor.QUESTION_KEYWORDS)):
                return self._generate_llm_response(query)
            
            else:
                # Default response for unhandled cases
                response = "I'm here to help with your insurance needs. Ready to apply or have specific questions?"
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
                return response
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "I apologize for the inconvenience. Please try asking your question again."


class UserInfoManager:
    """Manages user information operations"""
    
    REQUIRED_FIELDS = [
        ("interested_policy_type", "What type of insurance are you interested in (Term Life, Health, Investment, etc.)?"),
        ("age", "What's your age?"),
        ("desired_coverage", "What is the coverage amount you are looking for?"),
        ("premium_budget", "What is your budget for annual premium?"),
        ("premium_payment_mode", "How would you prefer to pay premiums (Annually, Semi-Annually, Monthly)?"),
        ("preferred_add_ons", "Are there any benefits you want (like accidental death cover, waiver of premium, etc.)?"),
        ("has_dependents", "Do you have any dependents (e.g., spouse, children, parents)? (yes/no)"),
        ("policy_duration_years", "How long would you like the policy coverage to last (in years)?"),
        ("insurance_experience_level", "Have you purchased insurance before? (Beginner, Intermediate, Experienced)"),
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
                continue
        
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