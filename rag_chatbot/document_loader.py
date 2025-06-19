# document_loader.py

import os
import fitz  # PyMuPDF kütüphanesi
from langchain.text_splitter import MarkdownHeaderTextSplitter

DOCS_PATH = 'company_docs'

def load_and_chunk_documents():
    """
    Belirtilen klasördeki tüm PDF'leri yükler, Markdown'a çevirir
    ve başlık hiyerarşisine göre parçalara ayırır.
    """
    all_chunks = []
    
    # Başlık seviyelerini ve onlara verilecek metadata isimlerini tanımla
    # Sizin dokümanınız "1. Başlık", "1.1. Alt Başlık" gibi ilerliyor.
    # Biz bunu # (Header 1) ve ## (Header 2) gibi düşüneceğiz.
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"), # Gerekirse daha fazla seviye ekleyebilirsiniz
    ]

    # Markdown ayırıcısını oluştur
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    for filename in os.listdir(DOCS_PATH):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(DOCS_PATH, filename)
            print(f"Processing document: {filename}")
            
            # 1. PDF'i aç ve metni çıkar
            doc = fitz.open(filepath)
            full_text = ""
            for page in doc:
                full_text += page.get_text()
            doc.close()

            # Not: Bu basit metin çıkarma. Dokümanınızın yapısına göre
            # başlıkları manuel olarak '#' ile işaretlemeniz gerekebilir.
            # Örneğin, "1. Services Provided by SEM" -> "# 1. Services Provided by SEM"
            # Bu dönüşümü otomatikleştirmek için regex kullanabilirsiniz, ama şimdilik manuel varsayalım.
            # Veya PDF'inizin bir Markdown versiyonunu oluşturun.
            # Şimdilik en basit haliyle ilerleyelim:
            
            # 2. Metni Markdown başlıklarına göre parçala
            chunks = markdown_splitter.split_text(full_text)
            
            # Her bir chunk'a kaynak dosya bilgisini ekle
            for chunk in chunks:
                chunk.metadata['source'] = filename
            
            all_chunks.extend(chunks)

    return all_chunks

# `chunk_documents` fonksiyonuna artık ihtiyacımız yok, çünkü yükleme ve parçalama birleşti.