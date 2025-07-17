#!/usr/bin/env python3
"""
Debug script to export all chunks and analyze specific queries
"""

import os
import sys
from pathlib import Path
import pickle
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_env_path = Path(__file__).parent / '.env'
load_dotenv(load_env_path)

from rag_chatbot.vector_store import FaissVectorStore
from rag_chatbot.embedding import LocalEmbeddingFunction

def export_chunks_to_file():
    """Export all chunks to chunks.txt for debugging"""
    
    # Check if vector store files exist
    vector_store_files = [
        'faiss_index.bin',
        'faiss_child_metadata.pkl',
        'faiss_parent_docstore.pkl'
    ]
    
    missing_files = [f for f in vector_store_files if not Path(f).exists()]
    if missing_files:
        print(f"❌ Missing vector store files: {missing_files}")
        print("Run the main application first to generate these files")
        return
    
    print("📁 Loading vector store...")
    
    # Initialize components
    embedding_model = LocalEmbeddingFunction()
    vector_store = FaissVectorStore(384)  # all-MiniLM-L6-v2 has 384 dimensions
    
    # Load existing vector store
    vector_store.load()
    
    print(f"📊 Total chunks in vector store: {len(vector_store.metadata)}")
    
    # Export chunks to file
    chunks_file = Path("chunks.txt")
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ALL CHUNKS EXPORT - FOR DEBUG PURPOSES ONLY\n")
        f.write("⚠️ WARNING: This file contains sensitive information - DELETE after use\n")
        f.write("="*80 + "\n\n")
        
        for i, chunk_doc in enumerate(vector_store.metadata):
            f.write(f"CHUNK {i+1}/{len(vector_store.metadata)}\n")
            f.write(f"Content: {chunk_doc.page_content}\n")
            f.write(f"Metadata: {chunk_doc.metadata}\n")
            f.write(f"Length: {len(chunk_doc.page_content)} chars\n")
            f.write("-" * 60 + "\n")
            f.write(f"CONTENT:\n{chunk_doc.page_content}\n")
            f.write("=" * 60 + "\n\n")
    
    print(f"✅ Chunks exported to: {chunks_file}")
    print(f"📄 Total chunks: {len(vector_store.metadata)}")
    
    # Calculate statistics
    total_chars = sum(len(chunk_doc.page_content) for chunk_doc in vector_store.metadata)
    avg_chunk_size = total_chars / len(vector_store.metadata) if vector_store.metadata else 0
    
    print(f"📊 Statistics:")
    print(f"   Total characters: {total_chars:,}")
    print(f"   Average chunk size: {avg_chunk_size:.0f} chars")
    print(f"   Estimated tokens: {total_chars * 0.222:.0f} tokens")
    
    return chunks_file

def debug_smartfeed_query():
    """Debug the SmartFeed query to understand why it uses 24000 tokens"""
    
    print("\n" + "="*80)
    print("DEBUGGING SMARTFEED QUERY")
    print("="*80)
    
    print("✅ Starting SmartFeed query analysis")
    
    # Test query
    query = "smartfeed nedir"
    print(f"🔍 Testing query: '{query}'")
    
    # Get relevant chunks (this is what the chatbot does internally)
    try:
        # Initialize embedding model and vector store
        embedding_model = LocalEmbeddingFunction()
        vector_store = FaissVectorStore(384)  # all-MiniLM-L6-v2 has 384 dimensions
        vector_store.load()
        
        # Get query embedding
        query_embedding = embedding_model.embed_documents([query])[0]
        
        # Search for relevant chunks with new optimized parameters
        relevant_docs = vector_store.search(query_embedding, top_k=5, score_threshold=0.4, max_context_length=15000)
        
        print(f"📊 Found {len(relevant_docs)} relevant documents")
        
        total_context_chars = 0
        total_context_tokens = 0
        
        print("\n📋 Document details:")
        for i, doc_content in enumerate(relevant_docs):
            doc_chars = len(doc_content)
            doc_tokens = int(doc_chars * 0.222)  # Approximate token count
            
            total_context_chars += doc_chars
            total_context_tokens += doc_tokens
            
            print(f"   Doc {i+1}: {doc_chars:,} chars, ~{doc_tokens:,} tokens")
            print(f"   Preview: {doc_content[:100]}...")
            print()
        
        print(f"📊 Total context:")
        print(f"   Characters: {total_context_chars:,}")
        print(f"   Estimated tokens: {total_context_tokens:,}")
        
        # Simulate full prompt
        system_prompt = "You are a helpful assistant for SEM company. Answer the user's question based on the provided context."
        user_query = query
        
        # Estimate full prompt tokens
        system_tokens = len(system_prompt) * 0.222
        query_tokens = len(user_query) * 0.222
        total_prompt_tokens = system_tokens + total_context_tokens + query_tokens
        
        print(f"📊 Full prompt estimation:")
        print(f"   System prompt: ~{system_tokens:.0f} tokens")
        print(f"   User query: ~{query_tokens:.0f} tokens")
        print(f"   Context: ~{total_context_tokens:.0f} tokens")
        print(f"   Total: ~{total_prompt_tokens:.0f} tokens")
        
        if total_prompt_tokens > 20000:
            print("🚨 HIGH TOKEN USAGE DETECTED!")
            print("Potential causes:")
            print("- Too many documents retrieved")
            print("- Documents are too large")
            print("- Low similarity threshold")
            print("- Overlapping or duplicate content")
        
        # Check for potential issues
        if len(relevant_docs) > 5:
            print(f"⚠️  Warning: Too many documents retrieved ({len(relevant_docs)})")
        
        if total_context_chars > 50000:
            print(f"⚠️  Warning: Context too large ({total_context_chars:,} chars)")
        
        # Show actual threshold used
        print(f"⚠️  Current similarity threshold: 0.25 (fallback: 0.15)")
        
    except Exception as e:
        print(f"❌ Error during query analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main function"""
    
    print("🚀 Starting chunk debug analysis...")
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please check your .env file")
        return
    
    try:
        # Export chunks
        export_chunks_to_file()
        
        # Debug SmartFeed query
        debug_smartfeed_query()
        
        print("\n" + "="*80)
        print("⚠️  REMEMBER TO DELETE chunks.txt AFTER ANALYSIS")
        print("⚠️  This file contains sensitive company information")
        print("="*80)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()