from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Tuple
import numpy as np
from tqdm import tqdm   
import pickle

from chunks import DocumentChunk

class RAGVectorStore:
    """FAISS-based vector store with advanced indexing"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_type: str = "IVFFlat"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.index_type = index_type
        self.index = None
        self.chunks = []
        self.chunk_metadata = {} 
        
    def create_vector_store(self, chunks: List[DocumentChunk], nlist: int = 100):
        """Create FAISS index with embeddings"""
        # logger.info(f"Creating vector index for {len(chunks)} chunks...")
        
        # Generate embeddings
        self._generate_embeddings(chunks)
        
        # Create embeddings matrix
        embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
        
        # Create FAISS index
        if self.index_type == "IVFFlat":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)  # Inner Product for cosine similarity
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            # Train the index
            self.index.train(embeddings)
        elif self.index_type == "HNSW":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:  # Flat index
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunks and metadata
        self.chunks = chunks
        self.chunk_metadata = {i: chunk for i, chunk in enumerate(chunks)}
        
        # logger.info(f"Index created successfully. Total vectors: {self.index.ntotal}")
    
    def _generate_embeddings(self, chunks: List[DocumentChunk]):
        """Generate embeddings for chunks"""
        # logger.info("Generating embeddings...")
        
        # Prepare texts for embedding
        texts = []
        for chunk in chunks:
            # Enhanced text for embedding (include metadata context)
            enhanced_text = chunk.content
            if chunk.metadata['section']:
                enhanced_text = f"Section: {chunk.metadata['section']}\n{enhanced_text}"
            texts.append(enhanced_text)
        
        # Generate embeddings in batches
        batch_size = 32
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,  # Important for cosine similarity
                show_progress_bar=False
            )
            
            # Assign embeddings to chunks
            for j, embedding in enumerate(batch_embeddings):
                chunks[i + j].embedding = embedding
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5, chunk_types: List[str] = None) -> List[Tuple[DocumentChunk, float]]:
        """Retrieve relevant chunks"""
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k * 2)  # Get more for filtering
        
        # Filter and rank results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                chunk = self.chunk_metadata[idx]
                
                # Filter by chunk type if specified
                if chunk_types is None or chunk.metadata["chunk_type"] in chunk_types:
                    results.append((chunk, float(score)))
        
        # Sort by score (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def hybrid_retrieve(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Tuple[DocumentChunk, float]]:
        """Hybrid retrieval combining dense and sparse methods"""
        # Dense retrieval
        dense_results = self.retrieve_relevant_chunks(query, k)
        
        # Simple sparse retrieval (keyword matching)
        sparse_results = self._sparse_retrieve(query, k)
        
        # Combine results
        combined_results = self._combine_retrieval_results(dense_results, sparse_results, alpha)
        
        return combined_results[:k]
    
    def _sparse_retrieve(self, query: str, k: int) -> List[Tuple[DocumentChunk, float]]:
        """Simple sparse retrieval using keyword matching"""
        query_tokens = set(query.lower().split())
        
        results = []
        for chunk in self.chunks:
            chunk_tokens = set(chunk.content.lower().split())
            
            # Calculate token overlap score
            intersection = query_tokens.intersection(chunk_tokens)
            union = query_tokens.union(chunk_tokens)
            
            if intersection:
                score = len(intersection) / len(union)
                results.append((chunk, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def _combine_retrieval_results(
        self, 
        dense_results: List[Tuple[DocumentChunk, float]], 
        sparse_results: List[Tuple[DocumentChunk, float]], 
        alpha: float
    ) -> List[Tuple[DocumentChunk, float]]:
        """Combine dense and sparse retrieval results"""
        # Create score maps
        dense_scores = {chunk.metadata["chunk_id"]: score for chunk, score in dense_results}
        sparse_scores = {chunk.metadata["chunk_id"]: score for chunk, score in sparse_results}
        
        # Get all unique chunks
        all_chunks = {}
        for chunk, _ in dense_results + sparse_results:
            all_chunks[chunk.metadata["chunk_id"]] = chunk
        
        # Calculate combined scores
        combined_results = []
        for chunk_id, chunk in all_chunks.items():
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            
            # Normalize scores (assuming dense scores are cosine similarity, sparse are overlap ratios)
            combined_score = alpha * dense_score + (1 - alpha) * sparse_score
            combined_results.append((chunk, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results
    
    def save_index(self, path: str):
        """Save the vector index and metadata"""
        index_path = f"{path}_faiss.index"
        metadata_path = f"{path}_metadata.pkl"
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata,
                'embedding_model_name': self.embedding_model.get_sentence_embedding_dimension(),
                'index_type': self.index_type
            }, f)
        
        # logger.info(f"Index saved to {index_path} and metadata to {metadata_path}")
    
    def load_index(self, path: str):
        """Load the vector index and metadata"""
        index_path = f"{path}_faiss.index"
        metadata_path = f"{path}_metadata.pkl"
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.chunk_metadata = data['chunk_metadata']
            self.index_type = data['index_type']
        
        # logger.info(f"Index loaded from {index_path}")

# class RAGVectorStore:
#     """Vector store management for document chunks"""
    
#     def __init__(self, persist_directory: str = "./chroma_db"):
#         self.embeddings = OpenAIEmbeddings()
#         self.persist_directory = persist_directory
#         self.vector_store = None
#         self.retriever = None
    
#     def create_vector_store(self, chunks: List[DocumentChunk]):
#         """Create vector store from document chunks"""
#         texts = [chunk.content for chunk in chunks]
#         metadatas = [chunk.metadata for chunk in chunks]
        
#         # Create Chroma vector store
#         self.vector_store = Chroma.from_texts(
#             texts=texts,
#             metadatas=metadatas,
#             embedding=self.embeddings,
#             persist_directory=self.persist_directory
#         )
        
#         # Create contextual compression retriever
#         base_retriever = self.vector_store.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": 15}
#         )
        
#         llm = OpenAI(temperature=0.1)
#         compressor = LLMChainExtractor.from_llm(llm)
#         self.retriever = ContextualCompressionRetriever(
#             base_compressor=compressor,
#             base_retriever=base_retriever
#         )
    
#     def retrieve_relevant_chunks(self, query: str, k: int = 10, 
#                                section_filter: Optional[str] = None) -> List[Document]:
#         """Retrieve relevant chunks for a query"""
#         if not self.retriever:
#             raise ValueError("Vector store not initialized")
        
#         # Add section filter if specified
#         if section_filter:
#             query = f"{query} section:{section_filter}"
        
#         return self.retriever.get_relevant_documents(query)
    
#     def hybrid_retrieve(self, query: str, k: int = 10) -> List[Document]:
#         """Hybrid retrieval with multiple strategies"""
#         if not self.vector_store:
#             return []
        
#         # Strategy 1: Similarity search with scores
#         similarity_results = self.vector_store.similarity_search_with_score(query, k=k*2)
        
#         # Strategy 2: MMR for diversity
#         mmr_results = self.vector_store.max_marginal_relevance_search(query, k=k)
        
#         # Combine and deduplicate results
#         all_results = []
#         seen_content = set()
        
#         # Add similarity results with scores
#         for doc, score in similarity_results:
#             if doc.page_content not in seen_content:
#                 doc.metadata['retrieval_score'] = score
#                 doc.metadata['retrieval_method'] = 'similarity'
#                 all_results.append(doc)
#                 seen_content.add(doc.page_content)
        
#         # Add MMR results
#         for doc in mmr_results:
#             if doc.page_content not in seen_content:
#                 doc.metadata['retrieval_method'] = 'mmr'
#                 all_results.append(doc)
#                 seen_content.add(doc.page_content)
        
#         return all_results[:k]