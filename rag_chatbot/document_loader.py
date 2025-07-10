import os
import re
import uuid

# Import PyMuPDF (the correct PDF library)
try:
    import pymupdf as fitz  # New PyMuPDF versions use this
except ImportError:
    try:
        import fitz  # Older PyMuPDF versions
        # Verify this is actually PyMuPDF by checking for Document class
        if not hasattr(fitz, 'Document'):
            raise ImportError("Wrong fitz package installed")
    except (ImportError, AttributeError):
        raise ImportError("PyMuPDF is required. Install with: pip install PyMuPDF")
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

DOCS_PATH = 'company_docs'

def load_and_chunk_documents():
    """
    Akıllı Mantıksal Gruplama Yöntemini Uygular:
    1. Her bir PDF'i tam bir metin olarak okur.
    2. Metni, "SECTION" ve "Case Studies" gibi ana mantıksal bloklara ayırır.
       Bu bloklar bizim "Parent Document"larımız (Ana Dokümanlar) olur.
    3. Her bir tam ve anlamlı Ana Dokümanı, aranabilir küçük "Child" (Çocuk) parçalara böler.
    """
    print("Starting Intelligent Logical Grouping process...")
    
    all_parent_documents = []
    
    # Her bir PDF dosyasını ayrı ayrı işleyeceğiz
    for filename in os.listdir(DOCS_PATH):
        if not filename.lower().endswith('.pdf'):
            continue
            
        filepath = os.path.join(DOCS_PATH, filename)
        print(f"--> Processing document: {filename}")
        
        # Open PDF document
        doc = fitz.open(filepath)
        try:
            full_text = "".join(page.get_text() for page in doc)
        finally:
            doc.close()

        # --- ANA MANTIKSAL BLOKLARI (PARENT) OLUŞTURMA ---

        # 1. Önce metni "SECTION" başlıklarına göre büyük parçalara ayıralım.
        # Bu, dokümanın genel yapısını korur. `(?=SECTION \d)` ifadesi ayıracı silmeden böler.
        sections = re.split(r'(?=SECTION \d:)', full_text, flags=re.IGNORECASE)
        
        for section_text in sections:
            if not section_text.strip():
                continue
                
            # Eğer bölüm "PROJECTS, SUCCESS STORIES" ise, onu daha da küçük mantıksal
            # parçalara, yani her bir vaka çalışmasına ayıralım.
            if "PROJECTS, SUCCESS STORIES" in section_text[:100]:
                # Vaka çalışmaları genellikle "• Ad (Kategori):" ile başlıyor. Bu bizim anahtarımız.
                # Geliştirilmiş regex: hem tek kelime hem de boşluklu markaları yakalar
                # Örnek: "• LC WAIKIKI (", "• Migros (", "• Boyner (" vs.
                case_studies = re.split(r'(?=\n•\s*[A-ZÇĞİİÖŞÜ][A-ZÇĞİİÖŞÜ\s]*\()', section_text)
                
                # İlk eleman genellikle bölüm başlığıdır, onu ayrı bir parent yapalım
                if case_studies[0].strip():
                    all_parent_documents.append(Document(page_content=case_studies[0].strip(), metadata={"source": filename}))
                
                # Geri kalan her bir vaka çalışmasını ayrı bir parent yapalım
                for study_text in case_studies[1:]:
                    if study_text.strip():
                        all_parent_documents.append(Document(page_content=study_text.strip(), metadata={"source": filename}))
            else:
                # Diğer bölümleri tek bir büyük parent olarak ekleyelim
                all_parent_documents.append(Document(page_content=section_text.strip(), metadata={"source": filename}))

    # --- ARANABİLİR ÇOCUK PARÇALARI (CHILD) OLUŞTURMA ---
    
    child_documents = []
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Smaller chunks for better precision
        chunk_overlap=100,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", ", ", " "]
    )
    
    for parent_doc in all_parent_documents:
        parent_id = str(uuid.uuid4())
        parent_doc.metadata['doc_id'] = parent_id
        
        # Ana dokümanın tam metninden küçük, aranabilir çocuk parçalar oluşturuyoruz
        smaller_chunks = child_splitter.split_text(parent_doc.page_content)
        
        for chunk_text in smaller_chunks:
            # Her çocuğa, ana dokümanın referansını ve metadatasını kopyalıyoruz
            child_metadata = parent_doc.metadata.copy()
            child_documents.append(Document(page_content=chunk_text, metadata=child_metadata))

    print(f"Total Parent Chunks (logical blocks) created: {len(all_parent_documents)}")
    print(f"Total Child Chunks for indexing created: {len(child_documents)}")
    print("Document loading and chunking finished successfully. 🚀")
    
    # Fonksiyonun çıktısı diğer dosyalarla uyumlu: (parent_list, child_list)
    return all_parent_documents, child_documents