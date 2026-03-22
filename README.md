# FlowML 🌊 

**Production-Ready Distributed AutoML Platform with GPU Acceleration**

FlowML is a privacy-first, self-hosted Automated Machine Learning (AutoML) platform designed to orchestrate complex data preprocessing, hyperparameter optimization, and distributed model training across a cluster of worker nodes. 

## 🏗️ System Architecture

FlowML operates on a split-compute, distributed architecture designed for scale and privacy:

### 1. The Core Application Stack
*   **Frontend**: React 18 with TypeScript and Vite. Uses Radix UI, Tailwind CSS, and Recharts for a highly interactive, metrics-driven dashboard. Handles dynamic ML problem switching (Regression vs. Classification).
*   **Backend Engine**: FastAPI built on asynchronous Python. Acts as the orchestration layer using SQLModel (SQLite/PostgreSQL) for persistence. Maintains WebSocket connections for real-time training telemetry.
*   **Message Broker & Caching**: Redis acts as the ephemeral datastore and message broker, linking the backend to the distributed workers.

### 2. The Distributed ML Workers (Celery)
*   **Intelligent Routing**: Workers probe their own underlying hardware on startup (CPU cores, RAM availability, GPU VRAM using pynvml, installed libraries). Tasks are routed via Celery queues (cpu, gpu, gpu.vram12, etc.) based on these capabilities.
*   **Job Execution**: Tasks like 	rain_automl and preprocess_dataset run decoupled from the API. The system gracefully falls back to background threading if Celery is unavailable.

### 3. The AutoML Pipeline (optuna_automl.py)
*   **Data Ingestion**: High-performance data parsing via Polars, supporting CSV, Parquet, Excel, JSON, and SQLite formats.
*   **Intelligent Preprocessing**: Automatically handles missing value imputation (median/mean), and cardinality-aware categorical encoding (One-Hot for low cardinality, Label Encoding for high cardinality).
*   **Hyperparameter Optimization (HPO)**: Powered by Optuna's TPESampler with strict 5-fold cross-validation. 
*   **Model Zoo**: 
    *   *Classification*: Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes.
    *   *Regression*: Ridge, Lasso, ElasticNet, SVR, Gradient Boosting, etc.
*   **Artifact Persistence**: Final models, label encoders, and preprocessor states are serialized and saved via joblib into the 	rained_models/ directory or an S3/MinIO compatible object store.

---

## 🚀 Quick Start

1. **Start the Database & Redis Requirements:**
   `ash
   docker-compose up -d redis postgres
   `
2. **Start the Backend:**
   `ash
   cd backend
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   uvicorn main:app --reload
   `
3. **Start the Celery Worker (In a separate terminal):**
   `ash
   cd backend
   celery -A worker.celery_app worker --loglevel=info -Q cpu,gpu
   `
4. **Start the Frontend:**
   `ash
   cd frontend
   npm install
   npm run dev
   `

---

## 🔒 Security & Privacy Focus
Unlike cloud-based AutoML solutions, FlowML never exports your datasets. Data ingestion, feature engineering, and inference all occur entirely within your controlled infrastructure. Supported deployment over Tailscale mesh networks allows for secure, multi-node clustering across geographic zones.
