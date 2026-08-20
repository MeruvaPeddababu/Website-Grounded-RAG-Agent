"""
Content processing - chunking and Gemini embedding generation.
"""
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langsmith import traceable

from src.crawler import PageContent


@dataclass
class Chunk:
    text: str
    url: str
    title: str
    chunk_index: int
    headings: list[str]
    token_count: int = 0


class ContentProcessor:
    """Process crawled pages into chunks with Gemini embeddings."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2",
            task_type="retrieval_document",
        )

    @traceable(
        name="chunking",
        run_type="tool",
        process_inputs=lambda i: {"pages": len(i["pages"])},
        process_outputs=lambda chunks: {
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "url": c.url,
                    "chunk_index": c.chunk_index,
                    "title": c.title,
                    "chars": len(c.text),
                    "preview": c.text[:300],
                }
                for c in chunks
            ],
        },
    )
    def process_pages(self, pages: list[PageContent]) -> list[Chunk]:
        """Split pages into chunks with metadata."""
        all_chunks = []

        for page in pages:
            header = f"Page: {page.title}\n"
            if page.headings:
                header += f"Sections: {' > '.join(page.headings[:5])}\n"
            header += "\n"

            full_text = header + page.text
            splits = self.text_splitter.split_text(full_text)

            for i, split in enumerate(splits):
                chunk = Chunk(
                    text=split,
                    url=page.url,
                    title=page.title,
                    chunk_index=i,
                    headings=page.headings,
                )
                all_chunks.append(chunk)

        print(f"Processed {len(pages)} pages -> {len(all_chunks)} chunks")
        return all_chunks

    def get_embeddings_model(self):
        return self.embeddings
