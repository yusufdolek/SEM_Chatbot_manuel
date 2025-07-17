#!/usr/bin/env python3
"""
Token measurement tool for company documents using Gemini API
This script measures how many tokens the company documents consume when sent to Gemini API
"""

import os
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tiktoken

# Load environment variables
load_env_path = Path(__file__).parent / '.env'
load_dotenv(load_env_path)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def count_tokens_tiktoken(text):
    """Count tokens using tiktoken (approximation)"""
    try:
        # Use cl100k_base encoding which is similar to what most models use
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens with tiktoken: {e}")
        return 0

def count_tokens_gemini(text):
    """Count tokens using Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        print(f"Error counting tokens with Gemini API: {e}")
        return 0

def analyze_company_docs():
    """Analyze all company documents and calculate total tokens"""
    
    company_docs_dir = Path(__file__).parent / "company_docs"
    
    if not company_docs_dir.exists():
        print(f"Company docs directory not found: {company_docs_dir}")
        return
    
    pdf_files = list(company_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in company_docs directory")
        return
    
    print("="*60)
    print("COMPANY DOCUMENTS TOKEN ANALYSIS")
    print("="*60)
    
    total_tiktoken = 0
    total_gemini = 0
    total_chars = 0
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        print("-" * 40)
        
        # Extract text
        text = extract_text_from_pdf(pdf_file)
        char_count = len(text)
        
        if not text.strip():
            print("⚠️  No text extracted from this file")
            continue
        
        # Count tokens with different methods
        tiktoken_count = count_tokens_tiktoken(text)
        gemini_count = count_tokens_gemini(text)
        
        print(f"📄 Characters: {char_count:,}")
        print(f"🔢 Tokens (tiktoken): {tiktoken_count:,}")
        print(f"🤖 Tokens (Gemini API): {gemini_count:,}")
        
        # Calculate ratios
        if char_count > 0:
            print(f"📊 Tiktoken ratio: {tiktoken_count/char_count:.3f} tokens/char")
            print(f"🎯 Gemini ratio: {gemini_count/char_count:.3f} tokens/char")
        
        total_chars += char_count
        total_tiktoken += tiktoken_count
        total_gemini += gemini_count
    
    print("\n" + "="*60)
    print("TOTAL SUMMARY")
    print("="*60)
    print(f"📁 Total PDF files: {len(pdf_files)}")
    print(f"📄 Total characters: {total_chars:,}")
    print(f"🔢 Total tokens (tiktoken): {total_tiktoken:,}")
    print(f"🤖 Total tokens (Gemini API): {total_gemini:,}")
    
    if total_chars > 0:
        print(f"📊 Average tiktoken ratio: {total_tiktoken/total_chars:.3f} tokens/char")
        print(f"🎯 Average Gemini ratio: {total_gemini/total_chars:.3f} tokens/char")
    
    print("\n" + "="*60)
    print("COST ESTIMATION (Gemini 2.0 Flash)")
    print("="*60)
    
    # Gemini 2.0 Flash pricing (as of 2024)
    # Input: $0.075 per 1M tokens
    # Output: $0.30 per 1M tokens
    
    input_cost_per_million = 0.075
    output_cost_per_million = 0.30
    
    # Assume average response is 500 tokens
    avg_response_tokens = 500
    
    # Calculate cost per query
    input_cost_per_query = (total_gemini / 1_000_000) * input_cost_per_million
    output_cost_per_query = (avg_response_tokens / 1_000_000) * output_cost_per_million
    total_cost_per_query = input_cost_per_query + output_cost_per_query
    
    print(f"💰 Input cost per query: ${input_cost_per_query:.6f}")
    print(f"📤 Output cost per query: ${output_cost_per_query:.6f}")
    print(f"💸 Total cost per query: ${total_cost_per_query:.6f}")
    
    # Calculate monthly costs for different usage levels
    queries_per_day = [10, 50, 100, 500, 1000]
    
    print(f"\n📈 Monthly cost estimates:")
    for queries in queries_per_day:
        monthly_cost = total_cost_per_query * queries * 30
        print(f"   {queries:4d} queries/day: ${monthly_cost:.2f}/month")
    
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    if total_gemini > 100_000:
        print("⚠️  HIGH TOKEN USAGE DETECTED!")
        print("🔧 Consider these optimizations:")
        print("   • Implement more aggressive chunking")
        print("   • Use semantic search with higher thresholds")
        print("   • Implement query classification")
        print("   • Cache frequent responses")
        print("   • Use smaller context windows")
    
    if total_gemini > 500_000:
        print("🚨 VERY HIGH TOKEN USAGE!")
        print("   • Consider using a different model")
        print("   • Implement hybrid search (keyword + semantic)")
        print("   • Use document summarization")
        print("   • Implement query routing")

def test_single_query_tokens():
    """Test how many tokens a typical RAG query uses"""
    
    print("\n" + "="*60)
    print("SINGLE QUERY TOKEN TEST")
    print("="*60)
    
    # Load a sample document chunk
    company_docs_dir = Path(__file__).parent / "company_docs"
    pdf_files = list(company_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found for testing")
        return
    
    # Get first 2000 characters as sample context
    sample_text = extract_text_from_pdf(pdf_files[0])[:2000]
    
    # Create a typical RAG prompt
    system_prompt = """You are a helpful assistant for SEM company. Answer the user's question based on the provided context. If the information is not in the context, say you don't know."""
    
    user_query = "SEM ne iş yapar?"
    
    full_prompt = f"""System: {system_prompt}

Context: {sample_text}

User: {user_query}"""
    
    print(f"📝 Sample query: {user_query}")
    print(f"📄 Context length: {len(sample_text)} chars")
    
    # Count tokens for the full prompt
    tiktoken_count = count_tokens_tiktoken(full_prompt)
    gemini_count = count_tokens_gemini(full_prompt)
    
    print(f"🔢 Full prompt tokens (tiktoken): {tiktoken_count:,}")
    print(f"🤖 Full prompt tokens (Gemini): {gemini_count:,}")
    
    # Break down by components
    system_tokens = count_tokens_gemini(system_prompt)
    context_tokens = count_tokens_gemini(sample_text)
    query_tokens = count_tokens_gemini(user_query)
    
    print(f"\n📊 Token breakdown:")
    print(f"   System prompt: {system_tokens:,} tokens")
    print(f"   Context: {context_tokens:,} tokens")
    print(f"   User query: {query_tokens:,} tokens")
    print(f"   Total: {system_tokens + context_tokens + query_tokens:,} tokens")

if __name__ == "__main__":
    print("🚀 Starting token analysis...")
    
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please check your .env file")
        sys.exit(1)
    
    try:
        # Analyze all company documents
        analyze_company_docs()
        
        # Test single query
        test_single_query_tokens()
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Analysis complete!")