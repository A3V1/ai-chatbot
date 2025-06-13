# data_processing/embedding_generator.py
from sentence_transformers import SentenceTransformer
from .mysql_connector import get_mysql_data
from langchain.embeddings import HuggingFaceEmbeddings
import os

def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def get_embeddings(model, texts: list[str]) -> list[list[float]]:
    return model.encode(texts, convert_to_tensor=False).tolist()

def chunk_text(text, max_chars=500):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
   