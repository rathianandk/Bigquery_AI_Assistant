# InstaKnow: BigQuery- NVIDIA Expert System

This project is a sophisticated, end-to-end AI assistant specialized in NVIDIA technologies, built natively on Google Cloud's BigQuery. It leverages BigQuery's powerful vector search and ML capabilities to provide accurate, context-aware answers to developer questions about CUDA, GPU programming, and more.

The assistant runs in an interactive, notebook-style environment, providing a rich user interface with detailed metrics, source citations, and creative visualizations.

## 🏗️ Architecture Overview

The system follows a multi-stage pipeline, all orchestrated within the Google Cloud ecosystem:

1.  **Data Ingestion & Processing**: NVIDIA documentation and Stack Overflow Q&A data are ingested, processed, and stored in structured BigQuery tables.
2.  **Embedding Generation**: BigQuery's native `ML.GENERATE_EMBEDDING` function is used to create vector embeddings for the text data. These embeddings are stored in dedicated BigQuery tables, ready for vector search.
3.  **Secure Authentication**: A secure REST API, deployed as a Cloud Run service, provides temporary access tokens to authenticate the notebook user and authorize access to GCP resources.
4.  **Interactive UI**: The user interacts with the assistant through a Jupyter-like interface powered by `ipywidgets`. They can select predefined questions or ask their own.
5.  **Vector Search & Context Retrieval**: The user's question is converted into an embedding. A `VECTOR_SEARCH` query is then executed in BigQuery to find the most relevant documents from the knowledge base.
6.  **Answer Generation & Display**: The retrieved documents are passed as context to a generative model in Vertex AI (`gemini-2.0-flash`). The model generates a comprehensive answer, which is then displayed to the user along with performance metrics, and search results.
## 核心文件 (Core Files)

-   `nvidia_AI_assistant.py`: The main application file. It contains the logic for the interactive UI, search, and answer generation. **Run this file in a Jupyter-compatible environment to start the assistant.**
-   `setup_data_pipeline.py`: A utility script to create the necessary BigQuery dataset, tables, and remote models for the project.
-   `generate_stackoverflow_embeddings.py`: A script to ingest Stack Overflow data and generate embeddings.

## 🚀 Quick Start

### 1. Setup GCP Environment

Make sure you have a Google Cloud project with the necessary APIs enabled.

```bash
# Set your project ID
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Enable required services
gcloud services enable bigquery.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable run.googleapis.com
```

### 2. Install Dependencies

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 3. Run the Assistant

Open `nvidia_AI_assistant.py` in a Jupyter-compatible environment (like VS Code with the Jupyter extension or a Jupyter Notebook) and run all cells. The interactive UI will appear, allowing you to ask questions.

## 🏆 Key Features

-   **BigQuery-Native**: Leverages BigQuery for storage, embedding generation, and vector search, minimizing data movement and complexity.
-   **Interactive Experience**: A rich UI with predefined questions, performance metrics, and visualizations provides a user-friendly experience.
-   **Hybrid Search**: Combines knowledge from both official NVIDIA documentation and community-driven Stack Overflow answers.
-   **Secure by Design**: Uses a token-based authentication system to securely access GCP resources.
-   **Creative Visualizations**: Includes a unique cosine wave plot to visualize the similarity scores of search results.
