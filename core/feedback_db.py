import json
import os
import shutil
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

DB_PATH = "./data/interactions.db"
QUERY_VECTOR_PATH = "./data/query_chroma_db"


class FeedbackDB:
    def __init__(self, embeddings: FastEmbedEmbeddings = None, db_path: str = DB_PATH, vector_path: str = QUERY_VECTOR_PATH):
        self.db_path = db_path
        self.vector_path = vector_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

        self.embeddings = embeddings or FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.client = self._create_client()
        self.query_store = self._init_query_store()
        self._init_sqlite()

    def _create_client(self):
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        try:
            return chromadb.PersistentClient(path=self.vector_path, settings=settings)
        except Exception:
            if os.path.exists(self.vector_path):
                shutil.rmtree(self.vector_path, ignore_errors=True)
            return chromadb.PersistentClient(path=self.vector_path, settings=settings)

    def _init_query_store(self):
        return Chroma(
            client=self.client,
            collection_name="queriom_query_memory",
            embedding_function=self.embeddings
        )

    def _ensure_query_store(self):
        try:
            self.query_store = self._init_query_store()
        except Exception:
            self.client = self._create_client()
            self.query_store = self._init_query_store()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_sqlite(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS query_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence TEXT,
                    sources_json TEXT,
                    rating INTEGER DEFAULT 0,
                    user_correction TEXT
                )
            """)
            conn.commit()

    def log_query(self, question: str, answer: str, confidence: str, sources: List[Dict[str, Any]]) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO query_logs (question, answer, confidence, sources_json)
                VALUES (?, ?, ?, ?)
            """, (question, answer, confidence, json.dumps(sources)))
            conn.commit()
            return cursor.lastrowid

    def record_feedback(self, query_id: int, rating: int, correction: Optional[str] = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE query_logs 
                SET rating = ?, user_correction = ?
                WHERE id = ?
            """, (rating, correction, query_id))
            conn.commit()

            cursor.execute("SELECT question, answer, user_correction FROM query_logs WHERE id = ?", (query_id,))
            row = cursor.fetchone()

        if row and (rating == 1 or correction):
            question, answer, user_corr = row
            final_verified_answer = user_corr if user_corr else answer

            doc = Document(
                page_content=question,
                metadata={
                    "query_id": query_id,
                    "verified_answer": final_verified_answer,
                    "rating": rating
                }
            )
            try:
                self.query_store.add_documents([doc])
            except Exception:
                self._ensure_query_store()
                try:
                    self.query_store.add_documents([doc])
                except Exception:
                    pass

    def search_similar_queries(self, incoming_question: str, k: int = 3, threshold: float = 0.85) -> List[Dict[str, Any]]:
        try:
            matches: List[Tuple[Document, float]] = (
                self.query_store.similarity_search_with_score(incoming_question, k=k)
            )
        except Exception:
            self._ensure_query_store()
            try:
                matches = self.query_store.similarity_search_with_score(incoming_question, k=k)
            except Exception:
                return []

        similar_exemplars = []
        for doc, distance in matches:
            if distance <= threshold:
                similar_exemplars.append({
                    "past_question": doc.page_content,
                    "verified_answer": doc.metadata.get("verified_answer", ""),
                    "distance": round(distance, 3)
                })
        return similar_exemplars

    def get_stats(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM query_logs")
            total = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM query_logs WHERE rating = 1")
            positive = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM query_logs WHERE rating = -1")
            negative = cursor.fetchone()[0]
            return {"total": total, "positive": positive, "negative": negative}

    def export_dpo_dataset(self, output_path: str = "./data/dpo_feedback.jsonl") -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT question, user_correction, answer 
                FROM query_logs 
                WHERE rating = -1 AND user_correction IS NOT NULL AND user_correction != ''
            """)
            rows = cursor.fetchall()

        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for question, chosen, rejected in rows:
                f.write(json.dumps({"prompt": question, "chosen": chosen, "rejected": rejected}) + "\n")
                count += 1
        return count