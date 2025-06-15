🤖 AI PDF QueryBot

Built by Sushant Kumar Pandey | VIT Vellore | 21BCI0321

An interactive, intelligent chatbot that reads and answers questions based on the content of any PDF file uploaded by the user. Powered by LangChain, FAISS, HuggingFace Embeddings, and Groq LLM (LLaMA3).

🚀 Features
📄 Upload any PDF file and extract meaningful insights

🤖 Ask natural language questions related to the PDF

🧠 Uses LLaMA3-8B via Groq for fast, high-quality answers

🔍 Context-aware retrieval with FAISS vector store

📚 Embeddings by HuggingFace (BAAI/bge-small-en-v1.5)

⚡ Streamlit-powered real-time interaction

💬 Maintains chat history for seamless conversations

🧠 Smart "Out of Context" detection for fallback or escalation

🖼️ Demo UI Snapshot


📦 Installation & Run


1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/ai-pdf-chatbot.git
cd ai-pdf-chatbot

2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt

3️⃣ Set Environment Variable
Create a .env file with your Groq API key:

bash
Copy
Edit
GROQ_API_KEY=your_groq_api_key

4️⃣ Run the App
bash
Copy
Edit
streamlit run app.py


🧠 How It Works

📤 Upload PDF → Extract text using PyPDFLoader

✂️ Chunk the text → Using RecursiveCharacterTextSplitter

🧬 Generate embeddings → HuggingFace (bge-small-en-v1.5)

📦 Store in FAISS → Enables semantic retrieval

💬 Ask questions → Retrieved context + LLaMA3 → Smart answers!



⚙️ Tech Stack

Tool / Library	Purpose
LangChain	Prompting, chains, retrieval
Groq (LLaMA3)	Fast, high-quality LLM inference
HuggingFace	Sentence embeddings
FAISS	Efficient vector similarity search
PyPDFLoader	PDF parsing and text extraction
Streamlit	Interactive frontend for the app



🧩 Future Improvements


✅ Voice-to-text support

✅ Multi-PDF support

✅ Live agent fallback integration

✅ Deployment via Docker or HuggingFace Spaces




🙋‍♂️ About the Author

Sushant Kumar Pandey

📧 Email: sushantpandey6203@gmail.com

🌐 Portfolio: your-website.com

🏫 VIT Vellore | CSE | 21BCI0321


