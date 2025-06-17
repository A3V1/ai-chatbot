import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from data_processing.user_context import get_user_context, load_user_data, get_user_info, save_user_data
from jinja2 import Template
from data_processing.mysql_connector import get_mysql_connection

# Load env vars
load_dotenv()
os.environ["PINECONE_ENVIRONMENT"] = "aped-4627-b74a"

# Load Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

raw_template = """
You are an expert insurance advisor chatbot helping users choose and apply for the best insurance policies.

🎯 GOALS:
- Recommend policies based on user profile and context.
- Answer follow-up questions (e.g. claims, exclusions, brochure).
- Guide user through: Recommendation → More Info → Application → Payment.
- Keep each message < 40 words unless clarification is required.

📚 KNOWLEDGE BASE CONTEXT:
{{ retrieved_context }}

👤 USER PROFILE:
- Phone: {{ phone_number }}
- Interested Policy Type: {{ interested_policy_type or "Not specified" }}
- Age: {{ age or "Not specified" }}
- Coverage Needed: {{ desired_coverage or "Not specified" }}
- Budget: {{ premium_budget or "Not specified" }}
- Payment Mode: {{ premium_payment_mode or "Not specified" }}
- Add-ons Wanted: {{ preferred_add_ons or "None specified" }}
- Dependents: {{ has_dependents or "Not specified" }}
- Policy Duration: {{ policy_duration_years or "Not specified" }}
- Insurance Experience: {{ insurance_experience_level or "Not specified" }}

🗣 CHAT HISTORY:
{{ chat_history }}

❓ USER QUESTION:
{{ question }}

---

💡 RESPONSE RULES:

1. **Understand the user's intent** from current question + profile + history.

2. **If recommending a policy**, consider:
   - Match based on age, dependents, budget, and interest
   - Include: policy name, premium, sum assured, and one key feature
   - Example: “FamilyCare Premium covers ₹15L, ₹21,400/year. Great for families. Want to see brochure?”

3. **If user asks for a brochure**, respond:
   - “Here’s the brochure: {{ policy_brochure_url }}”

4. **If asked about claim process**, respond:
   - “Claims are processed via {{ claim_process }}. Want to know about exclusions or benefits?”

5. **If asked about exclusions**, use:
   - “Exclusions include: {{ exclusions }}. Want to proceed or compare policies?”

6. **If interested in applying**, respond:
   - “I can help start your application. Shall we proceed with your details?”

7. **If user says yes to application**, say:
   - “Starting your application. You’ll get a confirmation shortly. Do you want to make payment now?”

8. **If user wants to pay**, respond:
   - “Redirecting to secure payment portal. Anything else you’d like help with?”

if user asks about payment, say: to type yes or no to proceed with payment.
---

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

    def ask(self, query: str):
        user_context = get_user_context(self.phone_number) or self.user_context or {}
        user_info = self.user_info or get_user_info(self.phone_number) or {}

        # Retrieve relevant documents and extract their content
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
            "retrieved_context": retrieved_context  # Add retrieved context here
        }

        prompt_template = Template(raw_template)
        full_prompt = prompt_template.render(**merged_context)
        response = self.llm.invoke(full_prompt)
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)

        # 🔄 SMART CONTEXTUAL SUMMARY GENERATION
        summary_prompt = f"""
        In 2 plain sentence, summarize what the user is seeking help with in this chat. Do not include any formatting or instructions.
Chat Transcript:
{chat_history_str}
        """
        summary_response = self.llm.invoke(summary_prompt)
        context_summary = summary_response.content.strip()
        if context_summary and len(context_summary.split()) > 5:
            save_user_data(self.phone_number, context_summary)

        return response.content


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