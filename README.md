# 🌱 Smart Agriculture Advisor

An AI-powered agricultural advisory system that helps farmers make better farming decisions using Artificial Intelligence, Retrieval-Augmented Generation (RAG), and a multi-agent workflow.

The system provides crop recommendations, soil health analysis, pest and disease guidance, weather-based irrigation suggestions, and practical farm planning.

## 📌 Project Overview

Smart Agriculture Advisor is designed to provide farmers with useful and accessible agricultural information.

Farmers can enter their farm details or ask questions in natural language. The system processes the information using AI agents and retrieves relevant agricultural knowledge from reference documents to generate helpful answers.

## 🎯 Objectives

- Recommend suitable crops based on soil and environmental conditions.
- Analyze soil health and provide improvement suggestions.
- Identify possible pest and disease problems.
- Provide irrigation and weather-related guidance.
- Generate practical farm management plans.
- Answer agricultural questions using RAG.
- Make agricultural information easier to access.

## ✨ Key Features

### 1. Crop Recommendation
Recommends suitable crops based on soil type, soil pH, water availability, season, and location.

### 2. Soil Health Analysis
Analyzes soil information and provides suggestions for improving soil fertility.

### 3. Pest and Disease Analysis
Provides possible pest and disease guidance based on crop symptoms and growth stage.

### 4. Weather and Irrigation Advisory
Provides irrigation and weather-related farming suggestions based on available farm information.

### 5. Farm Planning
Combines the outputs of the agricultural agents to generate a practical farm management plan.

### 6. Natural Language Question Answering
Farmers can ask questions such as:
- How can I improve soil fertility?
- Why are my tomato leaves turning yellow?
- Which crops are suitable for red soil?
- How can I control pests naturally?

The RAG pipeline retrieves relevant agricultural information and uses an AI model to generate an answer.

## 🏗️ System Architecture

The project uses a multi-agent architecture integrated with a Retrieval-Augmented Generation pipeline.

```text
                 FARMER
                    |
                    v
             STREAMLIT FRONTEND
                    |
                    v
               FASTAPI BACKEND
                    |
          +---------+----------+
          |                    |
          v                    v
   MULTI-AGENT WORKFLOW     RAG PIPELINE
          |                    |
    +-----+------+        Document Retrieval
    |     |      |              |
    v     v      v              v
   Crop  Soil   Pest       SQLite Vector Store
          |                    |
          v                    v
   Weather & Irrigation   Gemini Embeddings
          |
          v
      Farm Planning
          |
          v
     GROQ AI MODEL
          |
          v
    GENERATED RESULTS
          |
          v
       FARMER
```

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Frontend interface |
| FastAPI | Backend API |
| Groq | AI text generation |
| Gemini Embeddings | Text embedding generation |
| RAG | Agricultural knowledge retrieval |
| SQLite | Local vector storage and caching |
| LangGraph | Multi-agent workflow |
| PyPDF | PDF document processing |
| Git and GitHub | Version control |

## 📂 Project Structure

```text
Smart-Agriculture-Advisor/
│
├── agents/
│   ├── agriculture_workflow.py
│   ├── farm_state.py
│   ├── crop_recommendation_agent.py
│   ├── soil_health_agent.py
│   ├── pest_disease_agent.py
│   ├── weather_irrigation_agent.py
│   └── farm_planning_agent.py
│
├── rag/
│   ├── embedding_utils.py
│   ├── ingest.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   ├── evaluate_retrieval.py
│   └── vector_store/
│
├── backend/
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── docs/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/gayathriravikumar10a-dotcom/Smart-Agriculture-Advisor.git
```

Navigate into the project:

```bash
cd Smart-Agriculture-Advisor
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.env` file in the project root directory.

Add your API keys:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

**Important:** Never upload your `.env` file or API keys to GitHub.

## 🚀 Running the Application

### Step 1: Start the Backend

Open a PowerShell terminal:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

### Step 2: Start the Frontend

Open another PowerShell terminal:

```powershell
python -m streamlit run frontend/app.py
```

Open the local Streamlit URL displayed in the terminal.

## 🔄 RAG Pipeline

The Retrieval-Augmented Generation pipeline improves agricultural question answering by using relevant information from agricultural reference documents.

The workflow includes:

1. **Document Ingestion:** Reads agricultural PDF documents.
2. **Text Chunking:** Splits documents into smaller text sections.
3. **Embedding Generation:** Converts text into numerical vectors using Gemini embeddings.
4. **Vector Storage:** Stores embeddings and document information in SQLite.
5. **Query Processing:** Converts the farmer's question into an embedding.
6. **Similarity Search:** Retrieves relevant agricultural information.
7. **Answer Generation:** Uses Groq to generate a natural-language answer based on the retrieved context.

## 🤖 Multi-Agent Workflow

The application contains five main agricultural agents:

| Agent | Responsibility |
|---|---|
| Crop Recommendation Agent | Suggests suitable crops |
| Soil Health Agent | Analyzes soil conditions |
| Pest and Disease Agent | Provides pest and disease guidance |
| Weather and Irrigation Agent | Provides irrigation suggestions |
| Farm Planning Agent | Combines reports into a farm plan |

The agents work together through a sequential workflow to generate a comprehensive agricultural advisory report.

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend welcome endpoint |
| GET | `/health` | Checks backend health |
| POST | `/analyze-farm` | Generates agricultural reports |
| POST | `/ask-agriculture` | Answers agricultural questions |

Interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## 🧪 Testing and Evaluation

Run the retrieval evaluation:

```bash
python -m rag.evaluate_retrieval
```

Run retrieval metric evaluation:

```bash
python -m rag.evaluate_retrieval_metrics
```

Run chunk metric evaluation:

```bash
python -m rag.evaluate_chunk_metrics
```

Run answer evaluation:

```bash
python -m rag.evaluate_answers
```

Run the project tests:

```bash
python -m pytest
```

## 🔐 Security

- API keys are stored in environment variables.
- The `.env` file is excluded from Git tracking.
- Local vector database files are excluded from Git tracking.
- API credentials should never be committed to the repository.

## 🌾 Expected Outcome

Smart Agriculture Advisor provides an AI-based platform where farmers can:

- Receive crop recommendations.
- Understand soil health.
- Get pest and disease guidance.
- Receive irrigation suggestions.
- Generate farm management plans.
- Ask agricultural questions in natural language.

## 🔮 Future Enhancements

- Integration with real-time weather services.
- Regional language support.
- Image-based crop disease detection.
- Location-based agricultural service recommendations.
- Mobile-friendly farmer interface.
- Integration with live soil and weather data.

## 👩‍💻 Contributors

Developed as an academic Smart Agriculture project.

## 📜 Disclaimer

The AI-generated recommendations are informational and should not replace professional agricultural advice. Farmers should consult qualified agricultural experts before making important fertilizer, pesticide, irrigation, or crop management decisions.
