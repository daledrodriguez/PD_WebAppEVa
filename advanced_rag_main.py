
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    prepare_model_for_kbit_training,
    PeftModel
)
import re
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import psutil
import logging
from torch.cuda.amp import autocast


"""#ADVANCED RAG MAIN"""

@dataclass
class RetrievedDocument:
    content: str
    score: float
    metadata: Dict[Any, Any] = None

class HybridVectorIndex:
    def __init__(self, documents: List[str], embeddings: np.ndarray):
        self.documents = documents
        self.dense_embeddings = embeddings
        tokenized_docs = [self._simple_tokenize(doc.lower()) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _simple_tokenize(self, text: str) -> List[str]:
        # Simple word tokenization using split() and basic cleaning
        # Remove punctuation except periods for maintaining sentence structure
        text = re.sub(r'[^\w\s\.]', '', text)
        # Split on whitespace
        words = text.split()
        return words

    def search(self, query: str, query_embedding: np.ndarray, k: int = 3) -> List[RetrievedDocument]:
        # Dense search
        dense_scores = np.dot(self.dense_embeddings, query_embedding.T).squeeze()

        # Sparse search using simple tokenization
        tokenized_query = self._simple_tokenize(query.lower())
        sparse_scores = self.bm25.get_scores(tokenized_query)

        # Combine scores (weighted sum)
        combined_scores = 0.7 * dense_scores + 0.3 * sparse_scores
        top_indices = np.argsort(combined_scores)[-k:][::-1]

        return [
            RetrievedDocument(
                content=self.documents[idx],
                score=combined_scores[idx]
            ) for idx in top_indices
        ]

class AdvancedCodeReviewRAG:
    def __init__(self, debug=True):
        self.debug = debug
        self._setup_models()

    def _debug_print(self, message: str):
        if self.debug:
            print(f"DEBUG: {message}")

    def _setup_models(self):
        self._debug_print("Starting model setup...")

        # Initialize CUDA device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._debug_print(f"Using device: {self.device}")

        # Set up quantization
        self._debug_print("Setting up quantization config...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )

        self._debug_print("Loading base model...")
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                r"C:\Users\ASUS\Desktop\Advanced Rag\Mistral\Mistral-7B-Instruct-v0.3",
                quantization_config=quantization_config,
                device_map="auto"  # Automatically map to available devices
            )
        except TypeError as e:
            self._debug_print(f"Error loading model with quantization: {str(e)}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                r"C:\Users\ASUS\Desktop\Advanced Rag\Mistral\Mistral-7B-Instruct-v0.3",
                device_map="auto"  # Fallback to basic configuration
            )

        self._debug_print("Preparing model for LoRA...")
        self.base_model = prepare_model_for_kbit_training(self.base_model)

        # Load LoRA weights
        model_path = r"C:\Users\ASUS\Desktop\Advanced Rag\Mistral\finetuned-mistral-lora\final_model"
        if not os.path.exists(model_path):
            self._debug_print(f"LoRA weights not found at {model_path}")
            self.model = self.base_model
        else:
            self._debug_print(f"Loading LoRA weights from {model_path}")
            self.model = PeftModel.from_pretrained(
                self.base_model,
                model_path,
                torch_dtype=torch.float16
            )

        self._debug_print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\ASUS\Desktop\Advanced Rag\Mistral\Mistral-7B-Instruct-v0.3")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self._debug_print("Loading embedding models...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.embedder.to(self.device)

        self._setup_knowledge_base()
        self._debug_print("Model setup complete!")


    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def _setup_knowledge_base(self):
        # Path to the knowledge base directory
        knowledge_base_dir = r"C:\Users\ASUS\Desktop\Advanced Rag\Mistral\txt_knowledge_bases"

        # Load only the pep-0008.txt file
        file_name = "pep-0008.txt"
        file_path = os.path.join(knowledge_base_dir, file_name)

        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                if content:
                    documents.append(content.strip())
                else:
                    self._debug_print(f"Warning: File {file_name} is empty.")
        except Exception as e:
            self._debug_print(f"Error loading file {file_name}: {str(e)}")

        if not documents:
            self._debug_print("No documents were loaded into the knowledge base. Please check the file paths and contents.")
            documents.append("PEP 8 is the style guide for Python code.")  # Fallback document

        self._debug_print("Creating vector index...")
        document_embeddings = self.embedder.encode(documents, batch_size=4, show_progress_bar=True, device=self.device)
        self.index = HybridVectorIndex(documents, document_embeddings)


    def review_code(self, code: str, task_description: str) -> str:
        self._debug_print("Starting code review...")
        try:
            # Pre-retrieval optimization
            queries = self._optimize_query(code)
            retrieved_docs = []

            # Multi-query retrieval
            for query in queries:
                query_embedding = self.embedder.encode([query])[0]
                docs = self.index.search(query, query_embedding)
                retrieved_docs.extend(docs)

            # Post-retrieval processing
            reranked_docs = self._rerank_documents(code, retrieved_docs)
            context = self._compress_context(code, reranked_docs[:5])

            # Generate review
            self._debug_print("Generating review...")
            return self.generate_review(code, context, task_description)

        except Exception as e:
            self._debug_print(f"Error during review: {str(e)}")
            raise
        

    def _optimize_query(self, code: str) -> List[str]:
        self._debug_print("Optimizing query...")
        prompt = f"Generate 3 search queries to find relevant Python best practices for reviewing this code:\n{code}"

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                temperature=0.7,
                do_sample=True
            )

        queries = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [q.strip() for q in queries.split("\n") if q.strip()]

    def _rerank_documents(self, query: str, documents: List[RetrievedDocument]) -> List[RetrievedDocument]:
        self._debug_print("Reranking documents...")
        pairs = [(query, doc.content) for doc in documents]
        rerank_scores = self.cross_encoder.predict(pairs)

        for doc, score in zip(documents, rerank_scores):
            doc.score = float(score)

        return sorted(documents, key=lambda x: x.score, reverse=True)

    def _compress_context(self, code: str, documents: List[RetrievedDocument]) -> str:
        self._debug_print("Compressing context...")

        # Combine the code and top documents, ensuring it stays within a limit
        context_limit = 3000  # Set a limit for the combined input length
        combined_context = f"Given this code:\n{code}\n\nAnd these Python best practices:\n"
        for doc in documents:
            if len(combined_context) + len(doc.content) > context_limit:
                break
            combined_context += doc.content + "\n"

        # Prompt for summarization
        prompt = (
            f"{combined_context}\n\n"
            "Extract and summarize only the most relevant best practices for reviewing this code."
        )

        # Truncate if necessary
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,  # Increase this as needed
                max_new_tokens=300,  # Limit the new tokens generated
                temperature=0.3,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


    def generate_review(self, code: str, context: str, task_description: str, max_length: int = 4096) -> str:
        pep_content = context  # Assuming context contains relevant PEP guidelines.

        prompt = (
            f"Reiterate {code} "
            f"Task Description: {task_description}\n"
            f"Code Solution:\n{code}\n\n"
            f"PEP Content:\n{pep_content[:2000]}\n\n"
            "Does this PEP content apply specifically to improving or guiding this task and code? "
            "Respond with the strenght and weaknesses based on the PEP standards.'\n\n"
            f"Based on these Python best practices and standards:\n{context}\n\n"
            "Please review this Python code and provide specific feedback addressing code quality, potential bugs, performance, and suggested improvements:\n\n"
            f"```python\n{code}\n```"
                   f"Your feedback must:\n"
        f"- highlight few strengths in the code solution by using PEP standards as basis and cite what PEP standard it adheres.\n"
        f"- highlight some weaknesses in the code solution by using PEP standards as basis and cite what PEP standard it does not adhere.\n"
        f"- Provide actionable suggestive steps on how to solve the weaknesses.\n"
        f"- Show the code implementation of the suggestive steps.\n"
        f"- Provide short explanation about why PEP standards you talked about is important to follow with a short concluding message. \n"
        f"Feedback:"
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory_usage(message=""):
    """Print current memory usage with optional message"""
    print(f"{message} Memory Usage: {get_memory_usage():.2f} MB")

print_memory_usage("Initial")


