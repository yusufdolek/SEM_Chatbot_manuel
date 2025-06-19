# document_loader.py

import os
import fitz  # PyMuPDF
import re    # Regular Expressions kütüphanesi
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_PATH = 'company_docs'

def convert_text_to_markdown_headers(text):
    """Düz metindeki başlık benzeri yapıları Markdown başlıklarına çevirir."""
    # Olası formatlama hatalarını temizle
    text = re.sub(r' \n', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)

    # Numaralı başlıkları ve "SECTION" başlıklarını bul ve Markdown'a çevir
    # Örnek: "1.2. Digital Marketing" -> "## 1.2. Digital Marketing"
    text = re.sub(r'^(SECTION \d+:.*?)$', r'# \1', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^(\d\.\d\.\d\..*?)$', r'#### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d\.\d\..*?)$', r'### \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d\.\d.*?)$', r'## \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d\..*?)$', r'# \1', text, flags=re.MULTILINE)
    return text

def load_and_chunk_documents():
    """PDF'leri yükler, Markdown'a çevirir ve hibrit yöntemle parçalar."""
    all_final_chunks = []
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    # İkinci, boyuta göre bölecek olan parçalayıcıyı tanımlıyoruz
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64
    )

    for filename in os.listdir(DOCS_PATH):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"Processing document: {filename}")
            
            with fitz.open(filepath) as doc:
                full_text = "".join(page.get_text() for page in doc)
            
            # ADIM 1: Metni, başlıkları tanıyacak şekilde Markdown'a çevir
            markdown_text = convert_text_to_markdown_headers(full_text)
            
            # ADIM 2: Önce başlığa göre büyük parçalara böl
            initial_chunks = markdown_splitter.split_text(markdown_text)
            
            # ADIM 3: Bu büyük parçaları, metadatasını koruyarak daha küçüklere böl
            for large_chunk in initial_chunks:
                smaller_chunks = text_splitter.split_documents([large_chunk])
                for small_chunk in smaller_chunks:
                    # Başlık metadatasını her küçük parçaya kopyala
                    small_chunk.metadata.update(large_chunk.metadata)
                    all_final_chunks.append(small_chunk)

    return all_final_chunks