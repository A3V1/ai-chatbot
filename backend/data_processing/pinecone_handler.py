import os
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from data_processing.mysql_connector import get_mysql_data
from langchain_core.documents import Document

# Set Pinecone environment to match your actual index region
os.environ["PINECONE_ENVIRONMENT"] = "aped-4627-b74a"  # Replace if needed

def build_page_content(row):
    return f"""Policy Name: {row[2]}
Policy Type: {row[3]}
Coverage Amount: ₹{row[5]}
Annual Premium: ₹{row[4]}
Sum Assured: ₹{row[6]}

Eligibility: {row[19]}
Entry Age: {row[12]}–{row[13]} years
Policy Term: {row[14]}–{row[15]} years
Renewability: {row[16]}

Key Features:
- {row[20]}
Add-ons: {row[22]}
Exclusions: {row[21]}

Claim Process: {row[17]}
Maturity Benefits: {row[10]}
Return on Maturity: {row[11]}
Waiting Period: {row[9]}
Grace Period: {row[23]}
Free Look Period: {row[24]}

Tax Benefits: {row[18]}
COVID-19 Coverage: {bool(row[31])}
Premium Payment Modes: {row[27]}

Customer Rating: {row[29]}/5
Contact Support: {row[30]}
Brochure: {row[26]}

Tags: {row[32]}
Created At: {row[33]}
Last Updated: {row[34]}
"""

def prepare_documents():
    mysql_data = get_mysql_data()
    documents = [
        Document(
            page_content=build_page_content(row),
            metadata={
                "id": row[0],
                "insurer_name": row[1],
                "policy_name": row[2],
                "policy_type": row[3],
                "premium": row[4],
                "coverage_amount": row[5],
                "sum_assured": row[6],
                "co_payment": row[7],
                "network_hospitals": row[8],
                "waiting_period": row[9],
                "maturity_benefits": row[10],
                "return_on_maturity": row[11],
                "entry_age_min": row[12],
                "entry_age_max": row[13],
                "policy_term_min": row[14],
                "policy_term_max": row[15],
                "renewability": row[16],
                "claim_process": row[17],
                "tax_benefits": row[18],
                "eligibility": row[19],
                "features": row[20],
                "exclusions": row[21],
                "add_ons_available": row[22],
                "grace_period": row[23],
                "free_look_period": row[24],
                "policy_brochure_url": row[26],
                "premium_payment_modes": row[27],
                "policy_status": row[28],
                "customer_rating": row[29],
                "contact_support": row[30],
                "covid19_coverage": row[31],
                "policy_tags": row[32],
                "created_at": row[33],
                "last_updated": row[34],
            }
        )
        for row in mysql_data
    ]
    return documents

def upload_to_pinecone(index_name="insurance-chatbot", namespace="default"):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # Consider upgrading to bge-small if needed
    documents = prepare_documents()
    index = pc.Index(index_name)

    # Optional: Clear existing data
    index.delete(delete_all=True, namespace=namespace)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embedding,
        text_key="page_content",
        namespace=namespace
    )

    vectorstore.add_documents(documents)
    return vectorstore
