import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from data_processing.user_context import get_user_context, load_user_data, get_user_info, save_user_data, update_user_context, set_selected_plan
from jinja2 import Template
from data_processing.mysql_connector import get_mysql_connection, get_policy_brochure_url

# Load env vars
load_dotenv()
os.environ["PINECONE_ENVIRONMENT"] = "aped-4627-b74a"

# Load Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

raw_template = """
You are an expert insurance advisor guiding users through: Recommendation → Application → Payment.

🎯 PRIMARY GOAL: Lead conversations toward successful policy purchase and payment.

📚 CONTEXT: {{ retrieved_context }}
🔄 CONVERSATION STATE: {{ conversation_state }}

👤 USER PROFILE:
- Phone: {{ phone_number }}
- Policy Interest: {{ interested_policy_type or "Not specified" }}
- Age: {{ age }} | Budget: {{ premium_budget }}
- Coverage: {{ desired_coverage }} | Dependents: {{ has_dependents }}
- Experience: {{ insurance_experience_level }}

🗣 CHAT HISTORY: {{ chat_history }}
❓ CURRENT QUESTION: {{ question }}

---

💡 RESPONSE STRATEGY (Keep under 40 words):

**IF STATE = 'start':**
- Give ONE policy recommendation only if it is available. in database otherwise say "No policy matches your needs."
- Use user profile to recommend a policy.

- Format: "PolicyName covers ₹X, ₹Y/year. [Key benefit]. Ready to see details?"

**IF STATE = 'recommendation_given':**
- Don't repeat recommendations
- Answer questions about the recommended policy
- Always push toward: "Ready to apply?"
- if no policy matches the requirement : "No policy matches your needs. "

**STEP 2 - BUILD INTEREST:**
- If asked about brochure: "Here's the brochure: {{ policy_brochure_url }}. This plan suits your profile perfectly. Want to apply?"
- If asked about claims: "Claims via {{ claim_process }}, 24hr processing. Ready to secure your family's future?"
- If asked about exclusions: "Exclusions: {{ exclusions }}. Still excellent coverage for your needs. Shall we proceed?"

**STEP 3 - CLOSE FOR APPLICATION:**
- Always end responses with application prompts:
  - "Ready to apply and secure this rate?"
  - "Shall we start your application?"
  - "Want to lock in this premium today?"

**STEP 4 - PAYMENT TRANSITION:**
- When user shows interest: "Perfect! I'll start your application. You'll get confirmation shortly. Ready for payment?"
- Create urgency: "This rate is valid today. Secure it now?"

**PERSUASION TECHNIQUES:**
- Use urgency: "Limited time offer", "Lock in this rate"
- Social proof: "Popular with families like yours"
- Risk mitigation: "Protect your family's future"
- Value emphasis: "Just ₹X/month for ₹Y coverage"

**ALWAYS END WITH ACTION:**
Every response must include one of these:
- "Want to apply?"
- "Ready to proceed?"
- "Shall we secure this for you?"
- "Ready for payment?"

📝 RESPONSE:

"""


class ChatBot:
    def __init__(self, phone_number: str):
        self.phone_number = phone_number
        self.llm = ChatOpenAI(
            model="google/gemini-2.0-flash-lite-001",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=0.8
        )
        self.vectorstore = PineconeVectorStore.from_existing_index(
            index_name="insurance-chatbot",
            embedding=embedding
        )
        self.retriever = self.vectorstore.as_retriever()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        context, chat_history = load_user_data(phone_number)
        self.user_context = context
        chat_history = chat_history or []
        for msg in chat_history:
            if msg.get("type") == "human":
                self.memory.chat_memory.add_user_message(msg["content"])
            elif msg.get("type") == "ai":
                self.memory.chat_memory.add_ai_message(msg["content"])

        self.user_info = get_user_info(phone_number)

    def get_conversation_state(self):
        # Retrieve conversation_state from user_context
        user_context = get_user_context(self.phone_number) or {}
        return user_context.get('conversation_state', 'start')

    def set_conversation_state(self, state):
        # Update conversation_state in user_context
        update_user_context(self.phone_number, {'conversation_state': state})

    def ask(self, query: str):
        # State machine logic
        state = self.get_conversation_state()
        user_context = get_user_context(self.phone_number) or self.user_context or {}
        user_info = self.user_info or get_user_info(self.phone_number) or {}
        lower_query = query.strip().lower()

        print(f"DEBUG: Current state: {state}, Query: '{lower_query}'")  # Add this for debugging

        # Payment step
        if state == 'awaiting_payment_confirmation':
            yes_variants = ['yes', 'y', 'proceed', 'ok', 'okay', 'sure', 'let\'s go', 'go ahead', 'continue', 'confirm', 'pay', 'payment']
            if lower_query.strip() in yes_variants:
                self.set_conversation_state('payment_initiated')
                self.memory.chat_memory.add_user_message(query)
                # Fetch selected plan from context
                from data_processing.user_context import get_selected_plan
                plan = get_selected_plan(self.phone_number)
                # You should fetch the amount from a trusted backend source, e.g., a plan table
                # For now, fallback to user_info budget if not found
                amount = self.user_info.get('premium_budget', 1000)
                response = {
                    'action': 'redirect_to_payment',
                    'plan': plan,
                    'amount': amount,
                    'message': 'Redirecting to payment page...'
                }
                self.memory.chat_memory.add_ai_message('Redirecting to payment page...')
                return response
            elif lower_query in ['no', 'n']:
                self.set_conversation_state('showing_details')
                self.memory.chat_memory.add_user_message(query)
                response = 'No problem! Feel free to ask any questions about the policy.'
                self.memory.chat_memory.add_ai_message(response)
                return response

        # After payment is initiated, allow user to re-initiate payment if they ask
        if state == 'payment_initiated':
            if any(keyword in lower_query for keyword in ['pay', 'payment', 'do payment', 'proceed to payment', 'i want to pay']):
                from data_processing.user_context import get_selected_plan
                plan = get_selected_plan(self.phone_number)
                amount = self.user_info.get('premium_budget', 1000)
                response = {
                    'action': 'redirect_to_payment',
                    'plan': plan,
                    'amount': amount,
                    'message': 'Redirecting to payment page...'
                }
                self.memory.chat_memory.add_ai_message('Redirecting to payment page...')
                return response
            # Otherwise, allow normal questions or reset
            self.memory.chat_memory.add_user_message(query)
            response = "Your payment is being processed. If you have any more questions or need further assistance, let me know!"
            self.memory.chat_memory.add_ai_message(response)
            return response

        # Application confirmation step
        if state == 'awaiting_application_confirmation':
            if lower_query in ['yes', 'y']:
                self.set_conversation_state('awaiting_payment_confirmation')
                self.memory.chat_memory.add_user_message(query)
                response = "Perfect! Starting your application now. You'll get confirmation shortly. Ready to make payment?"
                self.memory.chat_memory.add_ai_message(response)
                return response
            elif lower_query in ['no', 'n']:
                self.set_conversation_state('showing_details')
                self.memory.chat_memory.add_user_message(query)
                response = 'No worries! Let me know if you have other questions about the policy.'
                self.memory.chat_memory.add_ai_message(response)
                return response

        # Handle "Ready to see details?" response - THIS IS THE KEY FIX
        if state == 'recommendation_given' and lower_query in ['yes', 'y', 'ready', 'show details', 'see details']:
            self.set_conversation_state('showing_details')
            last_policy = self.extract_last_policy_from_history()
            if last_policy:
                set_selected_plan(self.phone_number, last_policy)  # Store selected plan
                brochure_url = get_policy_brochure_url(last_policy)
                if brochure_url:
                    response = f"Here's the {last_policy} brochure: {brochure_url}\n\nThis plan is perfect for your profile. Ready to apply and lock in this rate?"
                else:
                    response = f"{last_policy} offers:\n✓ ₹10L coverage\n✓ Cashless hospitals\n✓ No waiting period for accidents\n✓ Family floater option\n\nReady to apply?"
            else:
                response = "not available. Please ask about other policies or provide more details."
            
            self.memory.chat_memory.add_ai_message(response)
            return response

        # Handle application requests
        if lower_query in ['apply', 'i want to apply', 'proceed', 'proceed to application'] or \
           (state == 'showing_details' and lower_query in ['yes', 'y']):
            self.set_conversation_state('awaiting_application_confirmation')
            self.memory.chat_memory.add_user_message(query)
            response = "Excellent choice! I can start your application right now. Shall we proceed with your details?"
            self.memory.chat_memory.add_ai_message(response)
            return response

        # Handle payment requests
        if lower_query in ['i want to pay', 'payment', 'pay', 'proceed to payment']:
            self.set_conversation_state('awaiting_payment_confirmation')
            self.memory.chat_memory.add_user_message(query)
            response = "Ready to secure your policy! Type 'yes' to proceed with payment or 'no' to wait."
            self.memory.chat_memory.add_ai_message(response)
            return response

        # Handle brochure requests
        if 'brochure' in lower_query or 'details' in lower_query:
            self.set_conversation_state('showing_details')
            last_policy = self.extract_last_policy_from_history()
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

        # CRITICAL: Only allow LLM for initial recommendation or specific questions
        if state == 'start':
            # Let LLM handle first recommendation only
            self.set_conversation_state('recommendation_given')
        elif state in ['recommendation_given', 'showing_details', 'awaiting_application_confirmation', 'awaiting_payment_confirmation']:
            # For these states, handle specific questions but don't give new recommendations
            if not any(keyword in lower_query for keyword in ['claim', 'exclusion', 'benefit', 'hospital', 'coverage', 'premium', 'what', 'how', 'when', 'where', 'why']):
                # If it's not a specific question, don't go to LLM
                response = "I'm here to help with your WellShield Elite policy. Ready to apply or have specific questions?"
                self.memory.chat_memory.add_user_message(query)
                self.memory.chat_memory.add_ai_message(response)
                return response

        # Only reach LLM for initial recommendation or specific informational questions
        retrieved_docs = self.vectorstore.similarity_search(query, k=6)
        retrieved_context = "\n---\n".join([doc.page_content for doc in retrieved_docs])

        chat_history_str = ""
        for msg in self.memory.chat_memory.messages:
            if msg.type == "human":
                chat_history_str += f"User: {msg.content}\n"
            elif msg.type == "ai":
                chat_history_str += f"Bot: {msg.content}\n"

        merged_context = {
            **user_context,
            **user_info,
            "chat_history": chat_history_str,
            "question": query,
        }
        full_prompt = Template(raw_template).render(merged_context)
        response = self.llm.invoke(full_prompt)
        
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)

        return response.content

    def extract_last_policy_from_history(self):
        """Extract the last recommended policy from chat history"""
        policies = ["SmartGrowth ULIP", "InvestSmart ULIP Plan", "MediCare Secure Plan", 
                "CarePlus Comprehensive", "WellShield Elite", "HealthGuard Basic", 
                "FamilyCare Premium", "SecureTerm Plus", "LifeShield Essential", 
                "AutoProtect Comprehensive", "FutureWealth Builder", "HomeSecure Essentials", 
                "TotalHome Protection Plan", "AcciGuard Secure", "LifeShield Critical Care"]
        
        for msg in reversed(self.memory.chat_memory.messages):
            if msg.type == "ai":
                for policy in policies:
                    if policy.lower() in msg.content.lower():
                        return policy
        return None


def get_missing_user_info_fields(user_info: dict) -> list:
    required_fields = [
        ("interested_policy_type", "What type of insurance are you interested in (Term Life, Health, Investment, etc.)?"),
        ("age", "What’s your age?"),
        ("desired_coverage", "What is the coverage amount you are looking for?"),
        ("premium_budget", "What is your budget for annual premium?"),
        ("premium_payment_mode", "How would you prefer to pay premiums (Annually, Semi-Annually, Monthly)?"),
        ("preferred_add_ons", "Are there any benefits you want (like accidental death cover, waiver of premium, etc.)?"),
        ("has_dependents", "Do you have any dependents (e.g., spouse, children, parents)? (yes/no)"),
        ("policy_duration_years", "How long would you like the policy coverage to last (in years)?"),
        ("insurance_experience_level", "Have you purchased insurance before? (Beginner, Intermediate, Experienced)"),
    ]
    return [(field, question) for field, question in required_fields if user_info.get(field) in [None, "", "null"]]


def update_user_info(phone_number: str, field: str, value):
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute(f"UPDATE user_info SET {field} = %s WHERE phone_number = %s", (value, phone_number))
    conn.commit()
    cursor.close()
    conn.close()


if __name__ == "__main__":
    phone_number = input("Enter your phone number: ").strip()
    if not phone_number:
        print("Phone number is required to start the chat.")
        exit(1)

    user_info = get_user_info(phone_number)
    if not user_info:
        print("Creating new user entry...")
        conn = get_mysql_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_info (phone_number) VALUES (%s)", (phone_number,))
        conn.commit()
        cursor.close()
        conn.close()
        user_info = get_user_info(phone_number)

    while True:
        missing_fields = get_missing_user_info_fields(user_info)
        if not missing_fields:
            break
        for field, question in missing_fields:
            value = input(question + " ").strip()
            if value.lower() in ["exit", "quit"]:
                exit(0)
            update_user_info(phone_number, field, value)
        user_info = get_user_info(phone_number)

    bot = ChatBot(phone_number)
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
        response = bot.ask(user_input)
        print("Bot:", response)