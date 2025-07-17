# SEM Kurumsal Bilgi Chatbot'u

Bu proje, SEM şirketinin kurumsal dokümanları üzerinde Sorgu-Cevaplama (Q&A) yapabilen, **Retrieval-Augmented Generation (RAG)** mimarisine dayalı bir chatbot uygulamasıdır. Chatbot, Flask tabanlı bir web arayüzü üzerinden kullanıcılarla etkileşime geçer ve Google'ın Gemini Pro modelini kullanarak doğal dilde cevaplar üretir.

## 🚀 Projenin Amacı

Projenin temel amacı, şirket içi veya dışı kullanıcıların, SEM'in hizmetleri, teknolojileri, başarı hikayeleri ve operasyonel yapısı hakkındaki sorularına, sağlanan PDF dokümanlarına dayanarak anında, doğru ve kapsamlı cevaplar vermektir.

## 🛠️ Teknoloji ve Mimarisi

Proje, modern bir RAG (Retrieval-Augmented Generation) mimarisine dayanmaktadır. Her bir bileşenin belirli bir görevi vardır:

-   **Web Arayüzü:** `Flask`
-   **LLM (Büyük Dil Modeli):** `Google Gemini Pro`
-   **Embedding Modeli:** `all-MiniLM-L6-v2` (Sentence-Transformers)
-   **Vektör Veritabanı:** `FAISS` (Facebook AI Similarity Search)
-   **Doküman İşleme:** `PyMuPDF`, `LangChain`
-   **Dil:** `Python`

### Proje Dosya Yapısı
<pre> chatbot_env/
company_docs/
rag_chatbot/
    __pycache__/
    __init__.py
    chatbot.py
    document_loader.py
    embedding.py
    llm.py
    vector_store.py
static/
templates/
    index.html
.env
.gitignore
app.py
deneme.md
faiss_index.bin
faiss_metadata.pkl
README.md
requirements.txt
</pre>


---

## ⭐ En Kritik Konsept: Doküman Parçalama (Chunking)

Bu projenin başarısı, büyük ölçüde dokümanların nasıl parçalandığına (**chunking**) bağlıdır. LLM'ler, onlara doğrudan verilmeyen bilgiyi bilemezler. **Doğru bilgiyi bulup LLM'e sunmak, RAG sisteminin en önemli ve en zorlu görevidir.**

### Karşılaşılan Zorluk ve Çözüm

Başlangıçta, dokümanlar sadece sabit bir karakter sayısına göre bölündü. Bu yöntem iki temel soruna yol açtı:
1.  **Bağlam Kaybı:** Bir başlık altındaki önemli bilgiler, farklı chunk'lara dağılarak anlamsal bütünlüğünü yitirdi.
2.  **Devasa Chunk'lar:** `MarkdownHeaderTextSplitter` tek başına kullanıldığında, bir başlık altındaki tüm metni tek bir devasa chunk olarak aldı. Bu da "sem ne iş yapar" gibi genel sorularda on binlerce token'lık girdi maliyetine ve düşük alaka puanlarına neden oldu.

**Uygulanan Çözüm: Hibrit Parçalama Stratejisi**

Bu sorunu çözmek için `document_loader.py` dosyasında iki aşamalı, hibrit bir parçalama stratejisi geliştirildi:

1.  **Önce Bağlama Göre Böl:** PDF dokümanı, içindeki başlık yapıları (`1.`, `1.1.`, `SECTION 1` vb.) Regex ile tanınıp standart Markdown başlıklarına (`#`, `##`) dönüştürülür. Ardından `MarkdownHeaderTextSplitter` ile metin, başlıklarına göre büyük, bağlamı korunmuş parçalara ayrılır.
2.  **Sonra Boyuta Göre Böl:** Bu büyük parçaların her biri, `RecursiveCharacterTextSplitter` kullanılarak **512 karakterlik** daha küçük ve yönetilebilir chunk'lara bölünür. Bu işlem sırasında, her küçük chunk'a ait olduğu ana başlığın metadata'sı miras bırakılır.

Bu hibrit yaklaşım sayesinde, her bir chunk hem yönetilebilir boyuttadır (maliyet ve verimlilik için) hem de hangi başlığa ait olduğunu "bilir" (bağlam ve doğruluk için).

### Alaka Puanı ve Eşik Değeri

Arama doğruluğunu artırmak ve gereksiz LLM çağrılarını önlemek için **Kosinüs Benzerliği** (`IndexFlatIP`) tabanlı bir arama endeksi kullanılmıştır. Bu, 0 ile 1 arasında anlamlı bir "alaka puanı" üretir.
-   Kullanıcı sorgusuna verilen cevapların alaka puanı, belirlenen bir eşik değerinin (`score_threshold`) altında kalırsa, LLM'e boş `context` gönderilir. Bu, sistemin ilgisiz konularda "bilmiyorum" demesini sağlar ve maliyeti ciddi oranda düşürür.

---

## 📊 Token Analizi ve Maliyet Raporu

### **Mevcut Durum:**
- **Toplam Doküman:** 113,749 karakter
- **Toplam Token (Gemini API):** 25,215 token
- **Sorgu başına maliyet:** $0.002041
- **Aylık maliyet tahminleri:**
  - 10 sorgu/gün: $0.61/ay
  - 100 sorgu/gün: $6.12/ay
  - 1000 sorgu/gün: $61.23/ay

### **Tek Sorgu Analizi:**
- **Sistem prompt:** 37 token
- **Context (2000 karakter):** 399 token
- **Kullanıcı sorgusu:** 5 token
- **Toplam:** 441 token

**Token Analizi Scripti:** `test_token.py` dosyası ile detaylı analiz yapılabilir.

## 🚀 Optimizasyon Önerileri

### **1. Token Optimizasyonu (Öncelik 1):**
- **Chunk boyutu küçültme:** 400 → 300 karakter
- **Daha yüksek similarity threshold:** 0.25 → 0.35
- **Context window sınırı:** Maksimum 3 chunk kullan
- **Query classification:** Basit sorular için daha az context

### **2. Retrieval İyileştirmeleri:**
- **Hybrid search:** Semantic + keyword search
- **Query expansion:** Eş anlamlı kelimeler ekle
- **Document ranking:** Relevance score'a göre sıralama
- **Negative sampling:** İlgisiz dokümanları filtrele

### **3. Caching Stratejileri:**
- **Response caching:** Sık sorulan sorular için
- **Embedding caching:** Aynı query'ler için
- **Context caching:** Benzer dokümanlar için
- **Session-based caching:** Kullanıcı başına

### **4. Advanced RAG Teknikleri:**
- **Self-querying:** Query'yi kategorize et
- **Multi-hop reasoning:** Birden fazla doküman kullan
- **Query rewriting:** Sorguyu optimize et
- **Contextual compression:** Gereksiz bilgileri çıkar

## 💻 Kod Geliştirmeleri

### **1. Performans Optimizasyonu:**
```python
# Async processing
async def process_multiple_queries()
# Connection pooling
# Background indexing
# Lazy loading
```

### **2. Hata Yönetimi:**
```python
# Retry mechanism
# Graceful degradation
# Fallback responses
# Health checks
```

### **3. Monitoring & Analytics:**
```python
# Query analytics
# Token usage tracking
# Performance metrics
# User behavior analysis
```

### **4. Güvenlik İyileştirmeleri:**
```python
# Rate limiting
# Input validation
# SQL injection protection
# XSS prevention
```

## 🔧 Debug ve Test Araçları

### **Chunk Analizi:**
- **chunks.txt dosyası:** Tüm chunk'ları test amaçlı dışa aktarma
- **⚠️ Dikkat:** chunks.txt dosyası test sonrası silinmelidir (hassas bilgi içerebilir)
- **Kullanım:** Debug ve chunk kalitesi analizi için

---

## 🚀 Projeyi Çalıştırma

1.  **Depoyu Klonlayın:**
    ```bash
    git clone [repo-url]
    cd [repo-adı]
    ```

2.  **Sanal Ortam Oluşturun ve Aktive Edin:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux için
    # venv\Scripts\activate    # Windows için
    ```

3.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ortam Değişkenlerini Ayarlayın:**
    -   Proje ana dizininde `.env` adında bir dosya oluşturun.
    -   `.env` dosyasının içine Google Gemini API anahtarınızı ve aşağıdaki konfigürasyonu girin:
        ```env
        GEMINI_API_KEY="AIzaSy..."
        TOKENIZERS_PARALLELISM=false
        ```

5.  **Dokümanlarınızı Ekleyin:**
    -   Bilgi kaynağı olarak kullanılacak tüm PDF dosyalarınızı `company_docs` klasörünün içine koyun.

6.  **Uygulamayı Başlatın:**
    ```bash
    flask run
    ```
    Uygulama başladığında, `company_docs` klasöründeki dokümanları işleyerek `faiss_index.bin` ve `faiss_metadata.pkl` dosyalarını otomatik olarak oluşturacaktır. Bu işlem, doküman sayısına bağlı olarak birkaç dakika sürebilir. *Not: Mevcut bir veritabanını yeniden oluşturmak için bu iki dosyayı manuel olarak silmeniz gerekir.*

7.  **Arayüze Erişin:**
    -   Tarayıcınızı açın ve `http://127.0.0.1:5000` adresine gidin.

## 🔮 Gelecek İyileştirmeler

-   **İndeksleme Script'i:** Vektör veritabanı oluşturma sürecini, web uygulamasının başlangıcından ayırıp ayrı bir `build_index.py` script'ine taşımak, uygulamanın daha hızlı başlamasını sağlar.
-   **Sohbet Geçmişi:** Konuşmanın bağlamını hatırlayabilmesi için sohbet geçmişi (chat history) özelliği eklenebilir.
-   **Gelişmiş Retriever'lar:** Daha karmaşık sorgular için `ParentDocumentRetriever` veya `Self-Querying Retriever` gibi LangChain'in gelişmiş arama mekanizmaları entegre edilebilir.
-   **Kullanıcı Arayüzü:** Cevapların daha iyi formatlanması (Markdown render) ve "streaming" (cevapların kelime kelime gelmesi) gibi özelliklerle arayüz zenginleştirilebilir.