import os
import shutil
from typing import Any, Dict, List, Tuple
import chromadb
from chromadb.config import Settings
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

GROUNDED_SYSTEM_PROMPT = """You are Queriom, an intelligent document analysis assistant.
Answer the user's question accurately using ONLY the provided context snippets.

GUIDELINES:
1. Base your answer directly on the context excerpts below.
2. Direct, logical deductions from slogans, headers, terms, and statements present in the context are encouraged.
3. If the context snippets contain zero relevant information or mentions of the topic, reply EXACTLY with:
   "I couldn't find information about that in the uploaded documents."
4. Do not invent unrelated facts or speculate beyond the document scope.

{semantic_memory_section}

CONTEXT SNIPPETS:
{context}

USER QUESTION:
{question}
"""

REPORT_SYNTHESIS_PROMPT = """You are Queriom Executive Intelligence Engine. 
Your objective is to generate an in-depth, professional, multi-section report based on the user's natural language request, relying EXCLUSIVELY on the provided document excerpts.

USER INQUIRY / REPORT OBJECTIVE:
{report_prompt}

REPORT TYPE / PERSPECTIVE:
{report_type}

CONTEXT EXCERPTS FROM CORPUS:
{context}

FORMAT THE REPORT USING THIS STRUCTURE:
[Report Title Reflecting the Subject]
**Target Objective:** {report_prompt}  
**Classification:** Grounded Enterprise Briefing  

---

1. Executive Summary
[High-level overview synthesized from the documents covering core principles, findings, and implications]

2. Key Findings & Detailed Analysis
[Detailed structured analysis broken down into subheadings, tables, or numbered points. Directly cite source files and pages where applicable, e.g. (Source: policy.pdf, Page 3)]

3. Evidentiary Source Matrix
[A bulleted breakdown identifying key claims, figures, or rules matched against their originating document and page numbers]

4. Scope Limitations & Information Gaps
[Crucial for zero-hallucination compliance: Explicitly identify what was NOT mentioned or left unanswered in the uploaded files regarding the user's query]

5. Strategic Takeaways / Recommendations
[Clear, actionable conclusions grounded strictly in the provided text]

STRICT ACCURACY RULES:
- Never extrapolate figures, dates, or organizational rules not present in the excerpts.
- If the documents lack sufficient data to compile a report on this topic, state clearly: "INSUFFICIENT CONTEXT: The uploaded corpus does not contain enough data to compile this report."
"""

DIGEST_PROMPT = """Analyze these document passages and generate:
1. A concise 2-sentence executive summary of the document contents.
2. Exactly 3 distinct, specific questions a user could ask about this document.

Format:
SUMMARY: <summary>
QUESTIONS:
- <question 1>
- <question 2>
- <question 3>

EXCERPTS:
{sample_text}
"""


class RAGPipeline:
    def __init__(self, groq_api_key: str, model_name: str = "openai/gpt-oss-20b", persist_dir: str = "./data/chroma_db"):
        self.persist_dir = persist_dir
        self.embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        self.llm = ChatGroq(
            model_name=model_name,
            groq_api_key=groq_api_key,
            temperature=0.1
        )
        self.client = self._create_client()
        self.vector_store = Chroma(
            client=self.client,
            collection_name="queriom_knowledge_base",
            embedding_function=self.embeddings
        )
        self.bm25_retriever = None

    def _create_client(self):
        settings = Settings(anonymized_telemetry=False, allow_reset=True)
        try:
            return chromadb.PersistentClient(path=self.persist_dir, settings=settings)
        except Exception:
            if os.path.exists(self.persist_dir):
                shutil.rmtree(self.persist_dir, ignore_errors=True)
            return chromadb.PersistentClient(path=self.persist_dir, settings=settings)

    def initialize_index(self, documents: List[Document], reset: bool = True):
        if reset:
            try:
                self.client.delete_collection("queriom_knowledge_base")
            except Exception:
                pass

        self.vector_store = Chroma(
            client=self.client,
            collection_name="queriom_knowledge_base",
            embedding_function=self.embeddings
        )
        self.vector_store.add_documents(documents)
        self.bm25_retriever = BM25Retriever.from_documents(documents)

    def _hybrid_retrieve(self, query: str, k: int = 6) -> List[Document]:
        dense_docs = self.vector_store.similarity_search(query, k=k)
        sparse_docs = self.bm25_retriever.invoke(query) if self.bm25_retriever else []

        combined = []
        seen = set()

        for doc in dense_docs + sparse_docs:
            content_key = doc.page_content.strip()
            if content_key not in seen:
                seen.add(content_key)
                combined.append(doc)
                if len(combined) >= k:
                    break

        return combined

    def query(self, question: str, semantic_exemplars: List[Dict[str, Any]] = None, distance_threshold: float = 1.15) -> Dict[str, Any]:
        if not self.vector_store:
            return {"answer": "No indexed documents found.", "sources": [], "confidence": "None"}

        matches: List[Tuple[Document, float]] = self.vector_store.similarity_search_with_score(question, k=1)

        if not matches or matches[0][1] > distance_threshold:
            return {
                "answer": "I couldn't find information about that in the uploaded documents.",
                "sources": [],
                "confidence": "Low"
            }

        best_distance = matches[0][1]
        confidence = "High" if best_distance < 0.75 else ("Moderate" if best_distance < 1.00 else "Low")

        retrieved_docs = self._hybrid_retrieve(question, k=6)

        unique_contexts = []
        seen_texts = set()
        sources = []

        for doc in retrieved_docs:
            content = doc.page_content.strip()
            if content not in seen_texts:
                seen_texts.add(content)
                meta = doc.metadata
                page = meta.get("page", 1)
                src_name = meta.get("source", "Document")
                unique_contexts.append(f"[{src_name} - Page {page}]\n{content}")
                sources.append({
                    "source": src_name,
                    "page": page,
                    "content": content,
                    "distance": round(best_distance, 3)
                })

        memory_section = ""
        if semantic_exemplars:
            formatted_mem = [f"SIMILAR PAST USER QUESTION: {ex['past_question']}\nVERIFIED APPROVED ANSWER: {ex['verified_answer']}" for ex in semantic_exemplars]
            memory_section = "VERIFIED SIMILAR PAST INQUIRIES (FROM RLHF MEMORY):\n" + "\n\n".join(formatted_mem) + "\n"

        prompt = ChatPromptTemplate.from_template(GROUNDED_SYSTEM_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "semantic_memory_section": memory_section,
            "context": "\n\n---\n\n".join(unique_contexts),
            "question": question
        })

        return {
            "answer": answer.strip(),
            "sources": sources,
            "confidence": confidence,
            "matched_memory": semantic_exemplars if semantic_exemplars else []
        }

    def generate_report(self, report_prompt: str, report_type: str = "Executive Analysis", k: int = 12) -> Dict[str, Any]:
        if not self.vector_store:
            return {"report": "No documents have been indexed yet.", "sources": []}

        retrieved_docs = self._hybrid_retrieve(report_prompt, k=k)

        unique_contexts = []
        seen_texts = set()
        sources = []

        for doc in retrieved_docs:
            content = doc.page_content.strip()
            if content not in seen_texts:
                seen_texts.add(content)
                meta = doc.metadata
                page = meta.get("page", 1)
                src_name = meta.get("source", "Document")
                unique_contexts.append(f"[{src_name} - Page {page}]\n{content}")
                sources.append({
                    "source": src_name,
                    "page": page,
                    "content": content[:120] + "..."
                })

        prompt = ChatPromptTemplate.from_template(REPORT_SYNTHESIS_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        report_output = chain.invoke({
            "report_prompt": report_prompt,
            "report_type": report_type,
            "context": "\n\n---\n\n".join(unique_contexts)
        })

        return {
            "report": report_output.strip(),
            "sources": sources,
            "source_count": len(unique_contexts)
        }

    def generate_document_digest(self, documents: List[Document]) -> Dict[str, Any]:
        sample = "\n\n".join([d.page_content for d in documents[:4]])[:3500]
        prompt = ChatPromptTemplate.from_template(DIGEST_PROMPT)
        chain = prompt | self.llm | StrOutputParser()
        raw = chain.invoke({"sample_text": sample})

        summary = "Document corpus indexed into persistent database."
        questions = ["What are the primary topics covered?"]

        if "SUMMARY:" in raw and "QUESTIONS:" in raw:
            parts = raw.split("QUESTIONS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            extracted = [q.strip().lstrip("-*0123456789. ") for q in parts[1].strip().split("\n") if q.strip()]
            if extracted:
                questions = extracted

        return {"summary": summary, "suggested_questions": questions[:3]}