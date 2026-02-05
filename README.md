# AI-Powered BI Platform

An intelligent, full-stack Business Intelligence platform that transforms raw data into actionable insights using AI and **adaptive dashboards** that completely change based on WHO is using it.

## 🚀 Key Features

### 1. **🎯 Adaptive Dashboards**
   - **Business Users** → Revenue KPIs, Sales Trends, Customer Insights (Power BI style)
   - **Analytics Users** → Statistical Distributions, Correlations, Histograms (Jupyter style)
   - **ML Engineers** → Model Metrics, Confusion Matrix, Feature Importance (MLflow style)
   - **Auto-Detection**: System automatically detects your persona from uploaded data

### 2. **Automated Data Engineering**
   - **Smart Profiling**: Automatically detects schema, missing values, duplicates, and outliers.
   - **One-Click Cleaning**: Apply recommended fixes (fill missing, drop duplicates) or write custom cleaning formulas.
   - **Quality Scoring**: Instant data quality assessment with detailed health reports.

### 3. **AI-Driven Analytics**
   - **Executive Summaries**: Uses **Groq AI (Llama 3)** to generate natural language summaries of your dataset.
   - **Conversational Q&A**: Chat with your data! Ask questions like "What is the trend of sales over time?" and get AI-generated answers based on statistical analysis.
   - **Context-Aware Insights**: Automatically identifies anomalies, trends, and key relationships.

### 4. **Dynamic Dashboard**
   - **Auto-Generated Visualizations**: Instantly creates Bar, Line, Area, and Donut charts based on column types.
   - **Interactive KPIs**: Key metrics are automatically calculated and displayed.
   - **Global Filtering**: Filter all charts and metrics by Category or Date ranges.

### 5. **Multi-Session Support**
   - **Isolated Sessions**: Each analysis session is completely isolated
   - **Session Persistence**: Resume work from where you left off
   - **Easy Switching**: Switch between multiple analysis projects

## 🛠️ Tech Stack

- **Frontend**: React 19 (Vite), Recharts, CSS3
- **Backend**: FastAPI (Python), Uvicorn
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **AI/LLM**: Groq API (Llama 3 70B)
- **Storage**: Local filesystem with session persistence

## 📦 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Groq API Key

### One-Command Setup
```bash
# Install all dependencies
npm run install:all

# Start both frontend and backend
npm run dev
```

This starts:
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:8000

### Manual Setup

#### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Run Server
python -m uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### 3. Configuration
Create a `.env` file in `backend/` and add your keys:
```env
GROQ_API_KEY=your_groq_api_key_here
```

## 🖥️ Usage

1.  **Upload**: Drag & drop your CSV/Excel files.
2.  **Clean**: Review quality issues and apply fixes.
3.  **Analyze**: Move to the Dashboard to see auto-generated insights.
4.  **Explore**: Use the "Custom Builder" or "Python Plotter" tabs to dig deeper.
5.  **Chat**: Ask the AI assistant for specific details.

## 📄 License
MIT License
