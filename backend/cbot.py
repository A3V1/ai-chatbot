import os
import logging
from typing import Dict, Optional, Any, Union, List
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain

from data_processing.user_context import (
    get_user_context, load_user_data, get_user_info, save_user_data,
    update_user_context, set_selected_plan, get_selected_plan
)
from jinja2 import Template
from data_processing.mysql_connector import get_mysql_connection, get_policy_brochure_url

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ConversationState(Enum):
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
        self.openai_api_key = os.getenv("OPENROUTER_API_KEY")
        # self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = "google/gemini-flash-1.5-8b"  # LLM used for response generation
        self.openai_api_base = "https://openrouter.ai/api/v1"
        # self.gemini_api_base=""# OpenRouter API endpoint
        self.temperature = 0.8  # LLM creativity
        self.embedding_model = "all-MiniLM-L6-v2"  # For semantic search
        self.pinecone_index = "insurance-chatbot"  # Pinecone index name
        self.similarity_search_k = 4 # Number of docs to retrieve
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
    
    @classmethod
    def get_policy_for_type(cls, policy_type: str) -> Optional[str]:
        """Return a policy name matching the interested_policy_type, or None if not found."""
        type_map = {
            "health": [
                "MediCare Secure Plan", "CarePlus Comprehensive", "WellShield Elite", "HealthGuard Basic", "FamilyCare Premium"
            ],
            "vehicle": [
                "AutoProtect Comprehensive"
            ],
            "term life": [
                "SecureTerm Plus", "LifeShield Essential", "LifeShield Critical Care"
            ],
            "investment": [
                "SmartGrowth ULIP", "InvestSmart ULIP Plan", "FutureWealth Builder"
            ],
            "home": [
                "HomeSecure Essentials", "TotalHome Protection Plan"
            ],
            "accident": [
                "AcciGuard Secure"
            ]
        }
        key = (policy_type or '').strip().lower()
        for t, policies in type_map.items():
            if t in key:
                for p in policies:
                    if p in cls.POLICIES:
                        return p
        return None


class UserInputProcessor:
    """Processes and categorizes user input for intent detection (affirmative, negative, etc.)."""

    GENERIC_DETAIL_KEYWORDS = {
        'details', 'show details', 'policy details', 'see details', 'show me details.', 'brochure',
        'i want to see details', 'claim process', 'exclusions', 'benefits', 'coverage','give me details',
    }
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
    
    QUESTION_KEYWORDS = {
        'claim', 'exclusion', 'benefit', 'hospital', 'coverage',
        'premium', 'what', 'how', 'when', 'where', 'why', 'brochure',
        *GENERIC_DETAIL_KEYWORDS, 'show brochure'
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


class CustomPromptTemplate:
    
    RAW_TEMPLATE = """
You are an expert insurance advisor with a proven track record of helping customers find the perfect insurance coverage. Your mission is to guide users through a structured sales process: Recommendation → Details → Application → Payment.

=== CONTEXT & DATA ===
Retrieved Insurance Information: {{ retrieved_context }}

Current Conversation State: {{ conversation_state }}

User Profile:
• Phone: {{ phone_number }}
• Age: {{ age }} years | Annual Budget: ₹{{ premium_budget }}
• Desired Coverage: ₹{{ desired_coverage }}
• Policy Interest: {{ interested_policy_type or 'General Insurance' }}
• Dependents: {{ has_dependents }} | Experience: {{ insurance_experience_level }}
• Payment Mode: {{ premium_payment_mode or 'Annual' }}

Previous Conversation: {{ chat_history }}

Current User Query: {{ question }}

=== RESPONSE GUIDELINES ===

**TONE & STYLE:**
- Professional yet friendly and conversational
- Confident and knowledgeable
- Solution-oriented with urgency
- Keep responses under 50 words unless providing detailed policy information
- Use bullet points for policy features, paragraphs for conversation

**CORE OBJECTIVE:** Every response must move the conversation toward policy purchase and payment completion.

=== STATE-BASED RESPONSE STRATEGY ===

**STATE: 'start'**
GOAL: Analyze user profile and recommend the most suitable policy immediately.

Response Format:
"Based on your profile , I recommend **{{policy_name}}**.
Key highlights:
• Coverage: ₹[amount]
• Premium: ₹[amount]/year
• [Top 2-3 benefits relevant to user]
 Would you like to see the complete details?"

Policy Selection Logic:
- Match user's interested_policy_type with available policies
- Consider age, budget, and coverage requirements
- If no exact match, recommend the closest suitable policy
- Always explain why this policy suits their specific needs

**STATE: 'recommendation_given'**
GOAL: Provide compelling details and move toward application. discuss with user recommend other options .

For Policy Details Request:
"**[Policy Name]** offers exceptional value for your profile:

✓ Coverage: ₹{{ desired_coverage }}
✓ Premium: ₹[amount]/{{ premium_payment_mode }}
✓ [Key benefit 1 - specific to user needs]
✓ [Key benefit 2 - competitive advantage]

Shall we start your application?"

For Brochure Request:
"Here's the detailed brochure: {{ policy_brochure_url }}
This policy is specifically designed for someone with your profile 

**STATE: 'showing_details'**
GOAL: Address concerns and close for application.

=== RESPONSE ENHANCEMENT RULES ===

**Always Include:**
1. **Personalization**: Reference user's age, budget, or family situation
2. **Urgency**: "Lock in this rate", "Secure today", "Don't wait"
3. **Social Proof**: "95% customer satisfaction", "Trusted by millions"
4. **Clear Next Step**: Specific call-to-action in every response

**Never Do:**
- Give generic responses without user context
- Provide multiple policy options (creates confusion)
- Use technical jargon without explanation
- End responses without a clear call-to-action
- Repeat the same information unnecessarily

=== QUALITY CONTROL ===

Before responding, ensure:
✓ Response directly addresses the user's current state and query
✓ Includes personalized elements from user profile
✓ Has a clear, compelling call-to-action
✓ Maintains sales momentum toward payment
✓ Uses appropriate tone for the conversation stage
✓ Stays within word limit unless providing detailed policy info

Remember: Your goal is not just to inform, but to guide the user to a purchase decision with confidence and urgency while maintaining trust and professionalism.
"""

prompt = PromptTemplate(
    template=CustomPromptTemplate.RAW_TEMPLATE,
    input_variables=[
        "retrieved_context", "conversation_state", "phone_number", "age",
        "premium_budget", "desired_coverage", "interested_policy_type",
        "has_dependents", "insurance_experience_level", "premium_payment_mode",
        "chat_history", "question", "context_summary",
        "context"
    ]
)

class ChatBot:
    def __init__(self, phone_number: str, initial_context: Optional[Dict[str, Any]] = None, initial_chat_history: Optional[List[Dict[str, Any]]] = None):
        self.phone_number = phone_number
        self.config = ChatBotConfig()
        self._initialize_components()
        
        if initial_context is not None and initial_chat_history is not None:
            self.user_context = initial_context
            self.user_info = get_user_info(self.phone_number) or {} # Always load latest user_info
            for msg in initial_chat_history:
                if msg.get("type") == "human":
                    self.memory.chat_memory.add_user_message(msg["content"])
                elif msg.get("type") == "ai":
                    self.memory.chat_memory.add_ai_message(msg["content"])
            logger.info(f"ChatBot initialized with provided context for phone: {self.phone_number}")
        else:
            self._load_user_data()        

    def _initialize_components(self) -> None:
        """Initialize LLM, vector store, retriever, and memory buffer."""
        try:
            self.llm = ChatOpenAI(
                model=self.config.model_name,
                openai_api_key=self.config.openai_api_key,
                openai_api_base=self.config.openai_api_base,
                temperature=self.config.temperature,
                # max_tokens=3000
            )
            
            embedding = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
            
            self.vectorstore = PineconeVectorStore.from_existing_index(
                index_name=self.config.pinecone_index,
                embedding=embedding
            )
            
            # Use ConversationBufferMemory for basic memory (no token counting)
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                # output_key="answer",  # ✅ Required
            )

            self.retriever = self.vectorstore.as_retriever()
            self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=False,
            output_key="answer",  # ✅ Required
            verbose=True,)
            
            self.chaintwo = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True,)
            
            logger.info("ChatBot components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ChatBot components: {e}")
            raise
    
    def add_user_plus_bot(self, user_message: str, bot_message: str) -> None:
        """
        Add both user and bot messages to the chat history in one call.
        """
        self.memory.chat_memory.add_user_message(user_message)
        self.memory.chat_memory.add_ai_message(bot_message)

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
        # Update context summary with
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
        logger.debug(f"[Payment Confirmation] Query: '{query}', Affirmative: {UserInputProcessor.is_affirmative(query)}, Negative: {UserInputProcessor.is_negative(query)}")
        if UserInputProcessor.is_affirmative(query):
            self.set_conversation_state(ConversationState.PAYMENT_INITIATED)
            logger.debug("User confirmed payment. Transitioning to PAYMENT_INITIATED and calling _handle_payment_initiated.")
            # Directly call payment initiation and return its response (which is PaymentResponse)
            response = 'payment initiated. Redirecting to payment page...'
            return response
        elif UserInputProcessor.is_negative(query):
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            response = 'No problem! Feel free to ask any questions about the policy.'
            self.add_user_plus_bot(query, response)
            logger.debug("User declined payment. Returning to SHOWING_DETAILS.")
            return response

    def _handle_payment_initiated(self, query: str) -> Union[PaymentResponse, str]:
        """Handle payment initiated state."""
        response = PaymentResponse(
            action='redirect_to_payment',
            plan=self.get_selected_plan(self.phone_number),
            amount=float(amount),
            message='Redirecting to payment page...'
        )
        # Add a button in the frontend when this message is received
        # The frontend should call GET /api/payment-details?phone_number=... when the button is clicked
        self.add_user_plus_bot(query, response.message)
        return response
    
    def _handle_application_confirmation(self, query: str) -> str:
        """Handle application confirmation state."""
        logger.info(f"[Application Confirmation] Received query: '{query}' (normalized: '{UserInputProcessor.normalize_input(query)}') for phone: {self.phone_number}")
        if UserInputProcessor.is_affirmative(query):
            response = "Great! I'm starting your application now. Please hold on while I process your details.Would you like to proceed to payment? Please reply with 'yes' or 'no'."

            self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
            # self.add_user_plus_bot(query, response)
            return response
        else:
            response = "No problem! If you have any questions or need more information about the policy, just let me know."
            self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
            # self.add_user_plus_bot(query, response)
            return response
    
    def _handle_recommendation_given(self, query: str) -> str:
        """Handle user input after a policy recommendation has been given."""
        if query.strip().lower() == "yes":
            self.set_conversation_state(ConversationState.SHOWING_DETAILS)
            return self._show_policy_details(query)
        else:
            # User does not want details, so recommend another policy (LLM response)
            return self._generate_llm_response(query)

    def _handle_application_request(self, query: str) -> str:
        """Handle user requests to apply for a policy."""
        self.set_conversation_state(ConversationState.AWAITING_APPLICATION_CONFIRMATION)
        response = "Excellent choice! I can start your application right now. Shall we proceed with your details?"
        self.add_user_plus_bot(query, response)
        return response
    
    def _handle_payment_request(self, query: str) -> str:
        """Handle user requests to proceed to payment."""
        self.set_conversation_state(ConversationState.AWAITING_PAYMENT_CONFIRMATION)
        response = "Ready to secure your policy!"
        self.add_user_plus_bot(query, response)
        return response
    
    def _show_policy_details(self, policy_name: str) -> str:
        # Handle generic queries like 'details', 'show details', etc.
        generic_keywords = {'details', 'show details', 'policy details', 'see details', 'show me details','brochure','i want to see details','claim_process','exclusions','benefits','coverage','premium','what','how','when','where','why'}
        if policy_name.strip().lower() in generic_keywords or policy_name not in PolicyManager.POLICIES:
            # Try to get last selected/recommended policy
            last_policy = self.get_selected_plan(self.phone_number)
            if not last_policy:
                last_policy = self._extract_last_policy_from_history()
            if last_policy:
                policy_name = last_policy
            else:
                # No policy found, reset state and prompt for recommendation
                self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
                return ("I couldn't find any policy details to show right now. "
                        "Would you like a recommendation? Please type 'recommend a policy' or specify the policy name you're interested in.")
        # Show details and move to application confirmation
        self.set_conversation_state(ConversationState.AWAITING_APPLICATION_CONFIRMATION)
        # Fetch policy details from database (describe insurance_policies)
        try:
            conn = get_mysql_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM insurance_policies WHERE policy_name = %s", (policy_name,))
            policy_row = cursor.fetchone()
            cursor.close()
            conn.close()
        except Exception as e:
            logger.error(f"Error fetching policy details from DB: {e}")
            self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
            return ("I couldn't find any policy details to show right now. "
                    "Would you like a recommendation? Please type 'recommend a policy' or specify the policy name you're interested in.")
        if not policy_row:
            self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
            return ("I couldn't find any policy details to show right now. "
                    "Would you like a recommendation? Please type 'recommend a policy' or specify the policy name you're interested in.")
        details = (
            f"Details for {policy_name}:\n"
            f"- Coverage Amount: ₹{policy_row.get('coverage_amount', 'Not specified')}\n"
            f"- Premium: ₹{policy_row.get('premium', 'Not specified')} per year\n"
            f"- Claim Process: {policy_row.get('claim_process', 'Standard claim process')}\n"
            f"- Exclusions: {policy_row.get('exclusions', 'Standard exclusions apply')}\n"
            f"- Add-ons Available: {policy_row.get('add_ons_available', 'Not specified')}\n"
            f"- Features: {policy_row.get('features', 'Not specified')}\n"
            f"- Eligibility: {policy_row.get('eligibility', 'Not specified')}\n"
            f"- Customer Rating: {policy_row.get('customer_rating', 'Not specified')}\n"
            f"- Policy Term: {policy_row.get('policy_term_min', 'Not specified')} to {policy_row.get('policy_term_max', 'Not specified')}\n"
            f"- Renewability: {policy_row.get('renewability', 'Not specified')}\n"
            f"- Tax Benefits: {policy_row.get('tax_benefits', 'Not specified')}\n"
            f"- Support: {policy_row.get('contact_support', 'Not specified')}\n\n"
            "Would you like to proceed with your application for this policy? (Type 'yes' to apply or 'no' to ask more questions.)\n"
        )
        logger.info(f"[Show Policy Details] Phone: {self.phone_number}, Policy: {policy_name}, State set to AWAITING_APPLICATION_CONFIRMATION")
        # Add bot message to chat history and save state after showing details
        self.add_user_plus_bot(policy_name, details)
        self._save_context()
        return details
        #add a line which takes input from user to proceed with application or ask more questions
        
        # If no details found in vectorstore, prompt for recommendation
        self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
        return ("I couldn't find any policy details to show right now. "
                "Would you like a recommendation? Please type 'recommend a policy' or specify the policy name you're interested in.")
        
    # def _generate_llm_response(self, query: str) -> str:
    #     try:
    #         # Run through the ConversationalRetrievalChain
    #         response= self.chain.invoke({"question": query})

    #         # Extract answer and add to memory
    #         bot_message = response.get("answer", "")
    #         self.add_user_plus_bot(query, bot_message)

    #         # Persist context and chat history
    #         self._save_context()

    #         return bot_message
            

    #     except Exception as e:
    #         logger.error(f"Error generating LLM response: {e}")
    #         return "Sorry, something went wrong while processing your request."
    
    def _generate_llm_response(self, query: str) -> str:
        try:
            if "recommend" in query.lower() and "policy" in query.lower():
                interested_policy_type = get_interested_policy_type(self.phone_number)
                search_query = query
                if interested_policy_type and "recommend" in query.lower():
                    search_query = f"{interested_policy_type} insurance policy recommendation {query}"
                elif interested_policy_type:
                    search_query = f"{interested_policy_type} insurance {query}"

                # --- Deterministic policy selection ---
                selected_policy = None
                if interested_policy_type:
                    selected_policy = PolicyManager.get_policy_for_type(interested_policy_type)
                # Fallback: try to extract from chat history
                if not selected_policy:
                    selected_policy = self._extract_last_policy_from_history()

                retrieved_docs = self.vectorstore.similarity_search(
                    search_query, k=self.config.similarity_search_k
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
                    "selected_policy": selected_policy or ""
                }
                # If selected_policy, inject it as interested_policy_type for prompt clarity
                if selected_policy:
                    merged_context["interested_policy_type"] = interested_policy_type
                    merged_context["policy_name"] = selected_policy
                    full_prompt = Template(CustomPromptTemplate.RAW_TEMPLATE).render(merged_context)
                    response = self.llm.invoke(full_prompt)
                    self.add_user_plus_bot(query, response.content)
                    # --- Set selected_plan as soon as a valid policy is mentioned in the AI response ---
                    last_policy = self._extract_last_policy_from_history()
                    if last_policy:
                        logger.info(f"[LLM Response] Phone: {self.phone_number}, Extracted Policy: {last_policy}")
                        result = set_selected_plan(self.phone_number, last_policy)
                        logger.info(f"set_selected_plan result: {result}")
                    return response.content

            # --- NORMAL MODE (simple flow like Code 2) ---
            response = self.chain.invoke({"question": query})
            bot_message = response.get("answer", "")
            self.add_user_plus_bot(query, bot_message)
            self._save_context()
            return bot_message

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return "Sorry, something went wrong while processing your request."
    
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
    
    def get_selected_plan(self, phone_number: str) -> Optional[str]:
        """
        Retrieve the selected policy (plan) for the user from user_context table.
        Returns the policy name/type if set, else None.
        """
        try:
            # Call the imported function, not the method itself
            return get_selected_plan(phone_number)
        except Exception as e:
            logger.error(f"Error fetching selected_plan for {phone_number}: {e}")
            return None

    # def ask_chain(self, query: str):
    #     return self.chain.invoke({"question": query})

    def ask(self, query: str) -> Union[str, PaymentResponse]:
        if not query.strip():
            return "Please provide a valid question or request."
        
        # Always reload user context and state before handling input
        self.user_context = get_user_context(self.phone_number) or {}
        self.user_info = get_user_info(self.phone_number) or {}
        
        state = self.get_conversation_state()
        logger.debug(f"Processing query in state: {state.value}")
        
        try:
            # --- Prioritize state-based handlers first ---
            if state == ConversationState.AWAITING_APPLICATION_CONFIRMATION:
                result = self._handle_application_confirmation(query)
            elif state == ConversationState.AWAITING_PAYMENT_CONFIRMATION:
                result = self._handle_payment_confirmation(query)
            elif state == ConversationState.PAYMENT_INITIATED:
                result = self._handle_payment_initiated(query)
            elif state == ConversationState.RECOMMENDATION_GIVEN:
                result = self._handle_recommendation_given(query)
            elif state == ConversationState.SHOWING_DETAILS:
                result = self._show_policy_details(query)
            elif state == ConversationState.START:
                self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
                result = self._generate_llm_response(query)
            # --- If not in a special state, use keyword-based routing ---
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.PAYMENT_KEYWORDS):
                result = self._handle_payment_request(query)
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.APPLICATION_KEYWORDS):
                result = self._handle_application_request(query)
            elif query.strip().lower() in UserInputProcessor.GENERIC_DETAIL_KEYWORDS:
                last_policy = self.get_selected_plan(self.phone_number) or self._extract_last_policy_from_history()
                if last_policy and state in [ConversationState.SHOWING_DETAILS, ConversationState.RECOMMENDATION_GIVEN]:
                    result = self._show_policy_details(query)
                else:
                    self.set_conversation_state(ConversationState.RECOMMENDATION_GIVEN)
                    result = ("I couldn't find any policy details to show right now. "
                              "Would you like a recommendation? Please type 'recommend a policy' or specify the policy name you're interested in.")
            elif UserInputProcessor.contains_keywords(query, UserInputProcessor.QUESTION_KEYWORDS):
                if state == ConversationState.SHOWING_DETAILS:
                    result = self._show_policy_details(query)
                else:
                    result = self._generate_llm_response(query)
            else:
                self.set_conversation_state(ConversationState.SHOWING_DETAILS)
                result = self._generate_llm_response(query)
        except Exception as e:
            import traceback
            logger.error(f"Error processing query: {e}\n{traceback.format_exc()}")
            result = "I'm here to help you secure your insurance. You can ask about policy details, say 'yes' to proceed, or type 'pay' to move to payment."
        # Persist context after every message
        self._save_context()
        return result


class UserInfoManager:
    REQUIRED_FIELDS = [
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
        """Update user information in database. Only allow fields in REQUIRED_FIELDS."""
        allowed_fields = {f for f, _ in cls.REQUIRED_FIELDS}
        if field not in allowed_fields:
            logger.error(f"Attempted to update disallowed field: {field}")
            return False
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
def get_missing_user_info_fields(user_info: dict) -> list:
    
    from cbot import UserInfoManager
    return UserInfoManager.get_missing_fields(user_info)


def update_user_info(phone_number: str, field: str, value: str) -> bool:
    from cbot import UserInfoManager
    return UserInfoManager.update_user_info(phone_number, field, value)


# --- Policy Type Questions (single source of truth) ---
POLICY_TYPE_QUESTIONS = {
    "term life": [
        {"field": "term_duration_years", "question": "For how many years do you want the term life coverage?"},
        {"field": "health_conditions", "question": "Do you have any existing health conditions? (yes/no, specify if yes)"},
        {"field": "smoker_status", "question": "Are you a smoker? (yes/no)"}
    ],
    "health": [
        {"field": "pre_existing_diseases", "question": "Do you have any pre-existing diseases? (yes/no, specify if yes)"},
        {"field": "hospital_preference", "question": "Do you have a preferred hospital or hospital network?"},
        {"field": "maternity_need", "question": "Do you need maternity coverage? (yes/no)"}
    ],
    "vehicle": [
        {"field": "vehicle_type", "question": "What type of vehicle do you want to insure? (Car/Bike/Other)"},
        {"field": "vehicle_year", "question": "What is the year of manufacture of your vehicle?"},
        {"field": "idv_preference", "question": "Do you have a preferred Insured Declared Value (IDV)? (yes/no, specify if yes)"}
    ],
    "investment": [
        {"field": "investment_horizon", "question": "What is your investment horizon (in years)?"},
        {"field": "risk_appetite", "question": "What is your risk appetite? (Low/Medium/High)"},
        {"field": "expected_returns", "question": "What are your expected returns or goals?"}
    ],
    "home": [
        {"field": "property_type", "question": "What type of property do you want to insure? (Apartment/House/Other)"},
        {"field": "property_value", "question": "What is the approximate value of your property?"},
        {"field": "natural_disaster_cover", "question": "Do you want natural disaster coverage? (yes/no)"}
    ]
}
# --- API/Frontend integration helpers ---

def get_policy_specific_questions(interested_policy_type: str) -> list:
    """
    Returns a list of dicts: [{"field": ..., "question": ...}] for the given policy type.
    For use by API/UI to fetch questions dynamically.
    """
    key = (interested_policy_type or '').strip().lower()
    return POLICY_TYPE_QUESTIONS.get(key, [])


def save_policy_specific_qa(phone_number: str, interested_policy_type: str, answers: dict):
    
    context, chat_history = load_user_data(phone_number)
    chat_history = chat_history or []

    questions = get_policy_specific_questions(interested_policy_type)
    
    qa_list = []
    for q in questions:
        field = q.get("field")
        question = q.get("question")
        answer = answers.get(field)
        qa_list.append({"field": field, "question": question, "answer": answer})
    chat_history.append({
        "type": "policy_specific_answers",
        "qa": qa_list
    })

    context_summary = context.get("context_summary") if context else ""
    save_user_data(phone_number, context_summary, chat_history)
    return True


def get_interested_policy_type(phone_number: str) -> str:

    _, chat_history = load_user_data(phone_number)
    chat_history = chat_history or []
    for msg in reversed(chat_history):
        if msg.get("type") == "interested_policy_type":
            return msg.get("content")
    return ""


def save_interested_policy_type(phone_number: str, interested_policy_type: str):
    
    # Load chat history
    context, chat_history = load_user_data(phone_number)
    chat_history = chat_history or []
    chat_history.append({
        "type": "interested_policy_type",
        "content": interested_policy_type
    })
    context_summary = context.get("context_summary") if context else ""
    save_user_data(phone_number, context_summary, chat_history)
    return True


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
            else:
                continue
        user_info = get_user_info(phone_number)

    interested_policy_type = get_interested_policy_type(phone_number)
    if not interested_policy_type:
        interested_policy_type = input("What type of insurance are you looking for? ").strip()
        if interested_policy_type.lower() in ["exit", "quit"]:
            return
        save_interested_policy_type(phone_number, interested_policy_type)

    # Load chat history from DB (or start new)
    context, chat_history = load_user_data(phone_number)
    chat_history = chat_history or []

    # Save updated chat_history to DB
    context_summary = context.get("context_summary") if context else ""
    save_user_data(phone_number, context_summary, chat_history)

    # Start chat
    print("\n=== Chat Started ===")
    print("Type 'exit' or 'quit' to end the conversation.\n")
    try:
        bot = ChatBot(phone_number)
        # # Immediately recommend a policy after collecting all info but only once if the state is START
        state = get_user_context(phone_number).get('conversation_state', ConversationState)
        # logger.info(f"[Main] Initial conversation state for phone {phone_number}: {state}")
        # # Only ask for recommendation if state is START and NOT already in application confirmation
        if state == ConversationState.START:
            initial_recommendation = bot.ask("recommend a policy")
            logger.info(f"[Main] Initial recommendation response type: {type(initial_recommendation)}")
            if isinstance(initial_recommendation, PaymentResponse):
                print(f"Bot: {initial_recommendation.message}")
                print(f"Action: {initial_recommendation.action}")
                print(f"Plan: {initial_recommendation.plan}")
                print(f"Amount: ₹{initial_recommendation.amount}")
            else:
                print(f"Bot: {initial_recommendation}")
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
