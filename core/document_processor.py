import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


class DocumentProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_file(self, uploaded_file) -> List[Document]:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()

        if file_ext == ".pdf":
            documents = self._extract_pdf(uploaded_file, file_name)
        elif file_ext == ".txt":
            documents = self._extract_txt(uploaded_file, file_name)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")

        raw_chunks = self.text_splitter.split_documents(documents)
        filtered_chunks = [c for c in raw_chunks if len(c.page_content.strip()) > 60]
        return filtered_chunks if filtered_chunks else raw_chunks

    def _extract_pdf(self, uploaded_file, file_name: str) -> List[Document]:
        docs = []
        reader = PdfReader(uploaded_file)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file_name,
                            "page": page_idx + 1,
                            "total_pages": len(reader.pages)
                        }
                    )
                )
        return docs

    def _extract_txt(self, uploaded_file, file_name: str) -> List[Document]:
        text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
        if not text.strip():
            return []
        return [
            Document(
                page_content=text,
                metadata={"source": file_name, "page": 1, "total_pages": 1}
            )
        ]