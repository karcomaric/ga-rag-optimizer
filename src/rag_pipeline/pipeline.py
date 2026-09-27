"""RAG pipeline based on LangChain and ChromaDB.

Thin glue layer over the standard building blocks: load the corpus, split,
persist embeddings, build retrievers. Chunkers, vector stores and retrievers
are taken from the libraries, none of them is reimplemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Hyperparameter configuration of a single pipeline variant.

    Chunk sizes count characters, ``embedding_model`` is a HuggingFace model ID
    and is ignored for ``retrieval_strategy="bm25"``. Valid strategies are
    ``dense``, ``bm25`` and ``hybrid``, checked in ``build_retriever``.
    """

    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model: str
    retrieval_strategy: str

    def __post_init__(self) -> None:
        """Rejects a chunk_overlap that is not strictly smaller than chunk_size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) muss kleiner sein als "
                f"chunk_size ({self.chunk_size})."
            )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Loads all ``.txt`` files of a corpus directory as Documents, sorted by name.

    Each file holds the title in line 1, a blank line, then the body text. The
    file stem becomes the ``paragraph_id`` in the metadata.
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus-Verzeichnis nicht gefunden: {corpus_dir}")

    documents: list[Document] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        lines = path.read_text(encoding="utf-8").splitlines()
        title = lines[0].strip() if lines else ""
        body = " ".join(line.strip() for line in lines[2:] if line.strip())
        documents.append(
            Document(
                page_content=body,
                metadata={"paragraph_id": path.stem, "title": title},
            )
        )
    return documents


def collection_name(config: PipelineConfig) -> str:
    """Builds the ChromaDB collection name from embedding model, chunk size and overlap."""
    return (
        f"{config.embedding_model.replace('/', '_')}_cs{config.chunk_size}_co{config.chunk_overlap}"
    )


def _split(documents: list[Document], config: PipelineConfig) -> list[Document]:
    """Splits the documents into chunks, which inherit the metadata of their source."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    return splitter.split_documents(documents)


_bm25_cache: dict[tuple[int, int, int], BM25Retriever] = {}


def _build_bm25(documents: list[Document], config: PipelineConfig) -> BM25Retriever:
    """Builds an in-memory BM25 retriever returning ``top_k`` chunks per query.

    Splitting and indexing produce the same retriever for every evaluation with
    identical chunking, so the result is kept per ``(chunk_size, chunk_overlap,
    top_k)`` for the lifetime of the process.
    """
    key = (config.chunk_size, config.chunk_overlap, config.top_k)
    if key not in _bm25_cache:
        chunks = _split(documents, config)
        _bm25_cache[key] = BM25Retriever.from_documents(chunks, k=config.top_k)
    return _bm25_cache[key]


_embeddings_cache: dict[str, HuggingFaceEmbeddings] = {}


def _get_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Returns the embedding model for ``model_name``, loading it on first use.

    Loading a sentence-transformers model takes seconds, which would otherwise
    be paid again on every single fitness evaluation.
    """
    if model_name not in _embeddings_cache:
        _embeddings_cache[model_name] = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache[model_name]


def _build_dense(
    documents: list[Document],
    config: PipelineConfig,
    chroma_dir: Path,
) -> BaseRetriever:
    """Builds a dense retriever with a ChromaDB cache per collection.

    An existing collection is reused as-is, because ``Chroma.from_documents``
    would append the same documents again and grow it on every call.
    """
    name = collection_name(config)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    embeddings = _get_embeddings(config.embedding_model)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = {c.name for c in client.list_collections()}
    if name in existing:
        logger.info("Collection %s aus dem Cache geladen", name)
        vectorstore = Chroma(
            client=client,
            collection_name=name,
            embedding_function=embeddings,
        )
    else:
        logger.info("Collection %s wird neu embeddet", name)
        vectorstore = Chroma.from_documents(
            documents=_split(documents, config),
            embedding=embeddings,
            collection_name=name,
            client=client,
            collection_metadata={"hnsw:space": "cosine"},
        )
    return vectorstore.as_retriever(search_kwargs={"k": config.top_k})


def _build_hybrid(
    documents: list[Document],
    config: PipelineConfig,
    chroma_dir: Path,
    weights: tuple[float, float] = (0.5, 0.5),
) -> Runnable[str, list[Document]]:
    """Builds a hybrid retriever from BM25 and dense, truncated to ``top_k``.

    The EnsembleRetriever fuses via reciprocal rank fusion and returns up to
    ``2*top_k`` documents, so a truncation step keeps the comparison with the
    single strategies fair. ``weights`` splits the fusion evenly by default.
    """
    bm25 = _build_bm25(documents, config)
    dense = _build_dense(documents, config, chroma_dir)
    ensemble = EnsembleRetriever(retrievers=[bm25, dense], weights=list(weights))
    top_k = config.top_k

    def retrieve(query: str) -> list[Document]:
        """Asks the ensemble and keeps only the first ``top_k`` documents."""
        results = ensemble.invoke(query)
        return results[:top_k]

    return RunnableLambda(retrieve)


def build_retriever(
    documents: list[Document],
    config: PipelineConfig,
    chroma_dir: Path,
) -> Runnable[str, list[Document]]:
    """Builds the retriever for the strategy in ``config``.

    Returns a Runnable with ``invoke(query) -> list[Document]``, the documents
    carry ``paragraph_id`` in ``metadata``. ``chroma_dir`` is unused for bm25.

    Raises:
        ValueError: If ``config.retrieval_strategy`` is not a known strategy.
    """
    if config.retrieval_strategy == "bm25":
        return _build_bm25(documents, config)
    if config.retrieval_strategy == "dense":
        return _build_dense(documents, config, chroma_dir)
    if config.retrieval_strategy == "hybrid":
        return _build_hybrid(documents, config, chroma_dir)
    raise ValueError(f"Unbekannte retrieval_strategy: {config.retrieval_strategy}")
