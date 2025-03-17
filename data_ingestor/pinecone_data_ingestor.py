from typing import List

from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

from config.config import Config
from file_loader.file_loader import File


CONTEXT_PROMPT = ChatPromptTemplate.from_template(
    """
You're an expert in document analysis. Your task is to provide brief, relevant context for a chunk of the text from the given document.

Here is the document:
<document>
{document}
</document>

Here is the chunk:
<chunk>
{chunk}
</chunk>

Provide a concise context (2-3 sentences) for this chunk, considering the following guidelines:
1. Identify the main topic or concept discussed in the chunk.
2. mention any relevent information or comparisons from broader document context.
3. If applicable, note how this info relates to the overall theme or purpose of the document.
4. Include any key figures, dates or percentages that provide important context.
5. Do not use phrases like "This chunk discisses" or "This section provides". Instead, directly state the main topic or concept.

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.

Context:
    """.strip()
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = Config.Preprocessing.CHUNK_SIZE,
    chunk_overlap = Config.Preprocessing.CHUNK_OVERLAP,
)


def create_llm() -> ChatOllama:
    return ChatOllama(
        model=Config.Preprocessing.LLM, 
        temperature=0, 
        keep_alive=-1,
    )

def create_embeddings() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings(
        model=Config.Preprocessing.EMBEDDING_MODEL
    )

def create_reranker() -> FlashrankRerank:
    return FlashrankRerank(
        model = Config.Preprocessing.RERANKER,
        top_n = Config.Chatbot.N_CONTEXT_RESULTS
    )

def _generate_context(llm: ChatOllama, document: str, chunk: str) -> str:
    messages = CONTEXT_PROMPT.format_messages(document=document, chunk=chunk)
    response = llm.invoke(messages)
    return response.content

def _create_chunks(document: Document) -> List[Document]:
    chunks = text_splitter.split_documents([document])
    if not Config.Preprocessing.CONTEXUALIZE_CHUNKS:
        return chunks
    llm = create_llm()
    contextual_chunks = []
    for chunk in chunks:
        context = _generate_context(llm, document.page_content, chunk.page_content)
        chunk_with_context = f"{context} \n\n {chunk.page_content}"
        contextual_chunks.append(Document(page_content=chunk_with_context, metadata=chunk.metadata))
    return contextual_chunks

def ingest_files(files: List[File]) -> BaseRetriever:
    documents = [Document(file.content, metadata={"source": file.name}) for file in files]
    chunks = []
    for document in documents:
        chunks.extend(_create_chunks(document))
    
    # Initialize Pinecone client
    pc = Pinecone(api_key=Config.VectorDB.PINECONE_API_KEY)
    
    # Create embeddings
    embeddings = create_embeddings()
    
    # Create vector store using langchain_pinecone package
    namespace = Config.VectorDB.PINECONE_NAMESPACE if Config.VectorDB.PINECONE_NAMESPACE else ""
    
    # Use from_documents to add new chunks
    vectorstore = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=Config.VectorDB.PINECONE_INDEX_NAME,
        namespace=namespace,
    )
    
    semantic_retriever = vectorstore.as_retriever(
        search_kwargs={"k": Config.Preprocessing.N_SEMENTIC_RESULTS}
    )

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Config.Preprocessing.N_BM25_RESULTS

    ensemble_retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    return ContextualCompressionRetriever(
        base_compressor=create_reranker(), 
        base_retriever=ensemble_retriever
    )