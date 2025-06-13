# import os
# from dotenv import load_dotenv
# import datetime


# # Load environment variables
# load_dotenv()

# # Set Pinecone environment to match your actual index region
# os.environ["PINECONE_ENVIRONMENT"] = "aped-4627-b74a"  # Set to your Pinecone environment/region

# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain.schema import Document
# from langchain.prompts import PromptTemplate
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_openai import ChatOpenAI
# from data_processing.mysql_connector import get_mysql_data

# def sanitize_metadata(metadata: dict) -> dict:
#     sanitized = {}
#     for key, value in metadata.items():
#         if value is None:
#             continue
#         if isinstance(value, (str, int, float, bool)):
#             sanitized[key] = value
#         elif isinstance(value, datetime.datetime):
#             sanitized[key] = value.isoformat()
#         elif isinstance(value, datetime.date):
#             sanitized[key] = value.strftime("%Y-%m-%d")
#         else:
#             # Skip complex types or convert to string if desired
#             sanitized[key] = str(value)
#     return sanitized

# # # --- Prepare Document objects from MySQL data ---
# def prepare_documents():
#     mysql_data = get_mysql_data()
#     documents = []
#     for row in mysql_data:
#         metadata = {
#             "id": row[0],
#             "insurer_name": row[1],
#             "policy_name": row[2],
#             "policy_type": row[3],
#             "premium": row[4],
#             "coverage_amount": row[5],
#             "sum_assured": row[6],
#             "co_payment": row[7],
#             "network_hospitals": row[8],
#             "waiting_period": row[9],
#             "maturity_benefits": row[10],
#             "return_on_maturity": row[11],
#             "entry_age_min": row[12],
#             "entry_age_max": row[13],
#             "policy_term_min": row[14],
#             "policy_term_max": row[15],
#             "renewability": row[16],
#             "claim_process": row[17],
#             "tax_benefits": row[18],
#             "eligibility": row[19],
#             "features": row[20],
#             "exclusions": row[21],
#             "add_ons_available": row[22],
#             "grace_period": row[23],
#             "free_look_period": row[24],
#             "policy_brochure_url": row[26],
#             "premium_payment_modes": row[27],
#             "policy_status": row[28],
#             "customer_rating": row[29],
#             "contact_support": row[30],
#             "covid19_coverage": row[31],
#             "policy_tags": row[32],
#             "created_at": row[33],
#             "last_updated": row[34]
#         }

#         sanitized_meta = sanitize_metadata(metadata)
#         documents.append(Document(page_content=row[25], metadata=sanitized_meta))
#     return documents

# # --- Load Embeddings ---
# embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# # --- Upload documents to Pinecone ---
# def upload_to_pinecone(index_name):
#     documents = prepare_documents()
#     vectorstore = PineconeVectorStore.from_documents(
#         documents=documents,
#         embedding=embedding,
#         index_name=index_name
#     )
#     return vectorstore

# # --- Create Prompt Template ---
# template = """
# You are an insurance agent. These humans will ask you questions about their insurance.
# Use the following piece of context to answer the question.
# If you don't know the answer, just say you don't know.
# Suggest the customer the right insurance policy according to their needs, giving enough information about it.
# try to keep the answer concise and to the point. andd try to answer in 60 words or less.

# Context: {context}
# Question: {question}
# Answer:
# """

# prompt = PromptTemplate(
#     template=template,
#     input_variables=["context", "question"]
# )

# # --- Define ChatBot class ---
# class ChatBot:
#     def __init__(self):
#         self.vectorstore = upload_to_pinecone("insurance-chatbot")
#         self.retriever = self.vectorstore.as_retriever()
#         self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
#         self.chain = ConversationalRetrievalChain.from_llm(
#             llm=ChatOpenAI(
#                 model="mistralai/mistral-small",
#                 openai_api_key=os.getenv("OPENAI_API_KEY"),
#                 openai_api_base="https://openrouter.ai/api/v1",
#                 temperature=0.8
#             ),
#             retriever=self.retriever,
#             memory=self.memory,
#             combine_docs_chain_kwargs={"prompt": prompt}
#         )

#     def ask(self, query):
#         return self.chain.invoke({"question": query})

# # --- Run the ChatBot session ---
# if __name__ == "__main__":
#     bot = ChatBot()
#     # Debug: test retrieval
#     docs = bot.retriever.get_relevant_documents("maternity benefits")
#     print(f">>> Retrieved {len(docs)} documents")
#     for doc in docs:
#         print(doc.metadata["policy_name"], doc.page_content[:100])
#     while True:
#         user_input = input("Ask me anything about insurance: ")
#         if user_input.lower() in ["exit", "quit"]:
#             break
#         result = bot.ask(user_input)
#         print(result)







import os
from dotenv import load_dotenv

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()
os.environ["PINECONE_ENVIRONMENT"] = "aped-4627-b74a"

# Load Embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Prompt Template
template = """
You are a helpful and concise virtual insurance advisor. Humans will ask about insurance policies. Use the conversation flow and context below to answer in **60 words or less**. If you don't know the answer, reply with "I don’t know."

Your goals:
- Ask follow-up questions according to the flow below to collect user needs.
- Recommend the most suitable policy type with a short explanation.
- Offer next actions clearly: Show brochure, Compare plans, Talk to human, or Buy now.

Use these conversation stages for context collection and recommendations:

 1. Basic Information
- What type of insurance are you looking for? (Store as: policy_type)
- What is your age? (age)
- Which city/state do you live in? (location)

 2. General Preferences
- Desired coverage amount? (coverage_amount)
- Maximum yearly premium budget? (premium_budget)
- Do you want a fixed sum assured? (sum_assured_preference)

 3. Policy-specific Questions (based on policy_type)
- Health: Ask about COVID-19 coverage, network hospitals, co-payments, pre-existing conditions, waiting period, add-ons, renewability, tax benefits.
- Term Life: Sum assured, term duration, maturity benefits, tax-saving, add-ons.
- Investment/ULIP: Market-linked returns, maturity benefit, lock-in, risk appetite, premium flexibility.
- Vehicle: Vehicle type, age, comprehensive or 3rd party, add-ons, fast claim preference.
- Home: Property value, natural disaster cover, ownership, belongings cover.

4. Post-Recommendation Actions
After suggesting a plan, offer:
- "Would you like to see the brochure or full policy details?"
- "Would you like to compare this plan with another?"
- "Would you like to proceed with purchasing this policy?"

5.At least ask 3 questions before making a recommendation. If the user provides all necessary information, you can skip to the recommendation.

6. if the user asks about a specific policy, provide a brief summary and ask if they want to see the brochure or compare it with others.

7.if the user wants to proceed with a purchase, exit the conversation and provide a confirmation message.

Context:
{context}

Chat History:
{chat_history}

User: {question}

Response:
"""

prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template
)

class ChatBot:
    def __init__(self):
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

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def ask(self, query, filter=None):
        # Use similarity search directly on vectorstore
        if filter is None:
            docs = self.vectorstore.similarity_search(query, k=4)
        else:
            docs = self.vectorstore.similarity_search(query, k=4, filter=filter)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Build chat history as string
        chat_history_str = ""
        for msg in self.memory.chat_memory.messages:
            if msg.type == "human":
                chat_history_str += f"User: {msg.content}\n"
            elif msg.type == "ai":
                chat_history_str += f"Bot: {msg.content}\n"

        # Create the full prompt using the PromptTemplate directly
        formatted_prompt = prompt.format(
            context=context,
            chat_history=chat_history_str,
            question=query
        )

        # Run the LLM on that prompt
        response = self.llm.invoke(formatted_prompt)

        # Update memory manually
        self.memory.chat_memory.add_user_message(query)
        self.memory.chat_memory.add_ai_message(response.content)

        return response.content

if __name__ == "__cbot__":
    bot = ChatBot()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = bot.ask(user_input)
        print("Bot:", response)