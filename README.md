# AI-Powered BI Platform

An intelligent, full-stack Business Intelligence platform that transforms raw data into actionable insights using AI and dynamic visualization tools.

## 🚀 Key Features

### 1. **Automated Data Engineering**
   - **Smart Profiling**: Automatically detects schema, missing values, duplicates, and outliers.
   - **One-Click Cleaning**: Apply recommended fixes (fill missing, drop duplicates) or write custom cleaning formulas.
   - **Quality Scoring**: Instant data quality assessment with detailed health reports.

### 2. **AI-Driven Analytics**
   - **Executive Summaries**: Uses **Groq AI (Llama 3)** to generate natural language summaries of your dataset.
   - **Conversational Q&A**: Chat with your data! Ask questions like "What is the trend of sales over time?" and get AI-generated answers based on statistical analysis.
   - **Context-Aware Insights**: Automatically identifies anomalies, trends, and key relationships.

### 3. **Dynamic Dashboard**
   - **Auto-Generated Visualizations**: Instantly creates Bar, Line, Area, and Donut charts based on column types.
   - **Interactive KPIs**: Key metrics are automatically calculated and displayed.
   - **Global Filtering**: Filter all charts and metrics by Category or Date ranges.

### 4. **Advanced Visualization Tools**
   - **Custom Chart Builder**: Drag-and-drop interface to create your own charts (Bar, Line, Scatter, etc.) with support for aggregations (Sum, Mean, Count) and automatic binning for large datasets.
   - **Python Plotter**: **[New]** A built-in code editor to execute custom Python code (using `matplotlib` and `seaborn`) directly in the browser for complex, publication-ready visualizations.

## 🛠️ Tech Stack

- **Frontend**: React.js (Vite), CSS3 (Custom Dashboard Design)
- **Backend**: FastAPI (Python), Uvicorn
- **Data Processing**: Pandas, NumPy
- **AI/LLM**: Groq API (Llama 3 70B)
- **Storage**: Local filesystem (Managed Uploads)

## 📦 Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 16+
- Groq API Key

### 1. Backend Setup
```bash
cd backend
# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
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
