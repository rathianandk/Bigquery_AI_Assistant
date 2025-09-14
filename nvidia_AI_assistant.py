
steps = """
const steps = [
    {
        title: "Step 1: Data Ingestion & Processing",
        text: "NVIDIA documentation and Stack Overflow Q&A data are ingested into BigQuery, where they are processed and stored in a structured format."
    },
    {
        title: "Step 2: Embedding Generation",
        text: "A BigQuery remote model is created to point to a specific Vertex AI model endpoint (e.g., text-embedding-005). BigQuery's native ML.GENERATE_EMBEDDING function then uses this remote model to create vector embeddings for the ingested text data, which are stored in BigQuery tables."
    },
    {
        title: "Step 3: Secure Authentication",
        text: "A temporary access token is fetched from a secure token service to authenticate and authorize access to Google Cloud Platform resources. This service is a REST API deployed to authenticate notebook users."
    },
    {
        title: "Step 4: User Asks a Question",
        text: "Run the notebook. The user asks a question through an interactive UI, which can be a predefined question or a custom one."
    },
    {
        title: "Step 5: Vector Search & Context Retrieval",
        text: "The user's question is converted into an embedding, and a VECTOR_SEARCH is performed in BigQuery to find the most relevant documents from the knowledge base."
    },
    {
        title: "Step 6: Answer Generation & Display",
        text: "The retrieved documents are used as context for a generative AI model in Vertex AI, which generates a comprehensive answer. The answer, along with performance metrics, is then displayed to the user in the UI."
    }
];
"""
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from google.oauth2 import credentials
from google.cloud import bigquery
from google.oauth2.credentials import Credentials
import vertexai
from vertexai.generative_models import GenerativeModel
import trafilatura # To extract content from URLs
import time
import threading
from functools import partial
import ipywidgets as widgets
from IPython.display import display, clear_output
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

class NVIDIAExpertSystem:
    def __init__(self):
        """Initialize with hardcoded configuration for judges"""
        # --- Hardcoded Configuration for Judges ---

        self.EMBEDDING_MODEL = "text_embedding_model"
        self.GEMINI_MODEL = "gemini-2.0-flash"
        self.CLOUD_RUN_TOKEN_URL = "https://bq-token-vendor-987726911762.us-central1.run.app/token"
        self.GCP_PROJECT_ID = "precise-mystery-466919-u5"
        self.DATASET_ID = "nvidia_docs_qa"
        self.EMBEDDING_MODEL = "text_embedding_model"
        self.GEMINI_MODEL = "gemini-2.0-flash"
         # Full table references
        self.NVIDIA_EMBEDDINGS_TABLE = f"`{self.GCP_PROJECT_ID}.{self.DATASET_ID}.unified_nvidia_embeddings`"
        self.NVIDIA_KNOWLEDGE_TABLE = f"`{self.GCP_PROJECT_ID}.{self.DATASET_ID}.unified_nvidia_knowledge`"
        self.SO_EMBEDDINGS_TABLE = f"`{self.GCP_PROJECT_ID}.{self.DATASET_ID}.stackoverflow_embeddings`"
        self.SO_KNOWLEDGE_TABLE = f"`{self.GCP_PROJECT_ID}.{self.DATASET_ID}.stackoverflow_knowledge_clone`"
              # self.answer_cache = {} # In-memory cache for generated answers
        self.EMBEDDING_MODEL_REF = f"`{self.GCP_PROJECT_ID}.{self.DATASET_ID}.{self.EMBEDDING_MODEL}`"

        # --- Fetch the Short-Lived Token ---
        print(f"🔑 Fetching temporary access token from: {self.CLOUD_RUN_TOKEN_URL}")
        try:
            resp = requests.get(self.CLOUD_RUN_TOKEN_URL)
            resp.raise_for_status()
            token = resp.json()["access_token"]
            print("✅ Successfully fetched temporary token!")
        except requests.exceptions.RequestException as e:
            print(f"❌ ERROR: Failed to get token. Details: {e}")
            raise

        # --- Initialize BigQuery Client with the Token ---
        print("📊 Initializing BigQuery client with temporary credentials...")
        creds = Credentials(token)
        self.client = bigquery.Client(credentials=creds, project=self.GCP_PROJECT_ID)
        print("✅ BigQuery client is ready.")

        # --- Initialize Vertex AI ---
        print("🚀 Initializing Vertex AI...")
        vertexai_creds = Credentials(token)
        vertexai.init(
            project=self.GCP_PROJECT_ID,
            location="us-central1",
            credentials=vertexai_creds
        )
        self.gen_model = GenerativeModel(self.GEMINI_MODEL)
        self.console = Console()
        print("✅ Vertex AI initialized!")

    def get_embeddings_from_bigquery(self, texts):
        """Get embeddings using configured embedding model"""
        embeddings = []
        for text in texts:
            safe_text = text.replace("'", "''").replace('"', '""')
            query = f"""
            SELECT ml_generate_embedding_result
            FROM ML.GENERATE_EMBEDDING(
                MODEL {self.EMBEDDING_MODEL_REF},
                (SELECT '{safe_text}' AS content)
            )
            """
            query_job = self.client.query(query)
            result = query_job.result()
            for row in result:
                embeddings.append(row.ml_generate_embedding_result)
        return embeddings

    def search_similar_documents(self, question, top_k=10):
        """Search for similar documents across NVIDIA and Stack Overflow sources."""
        question_embedding = self.get_embeddings_from_bigquery([question])[0]
        embedding_str = ','.join(map(str, question_embedding))

        # Query for NVIDIA documentation
        # Query for NVIDIA docs
        nvidia_query = f"""
            WITH search_results AS (
                SELECT
                    base.doc_id,
                    distance
                FROM VECTOR_SEARCH(
                    TABLE {self.NVIDIA_EMBEDDINGS_TABLE},
                    'embedding',
                    (SELECT [{embedding_str}] AS query_vector),
                    top_k => {top_k},
                    distance_type => 'COSINE'
                )
            )
            SELECT
                'NVIDIA Docs' AS source_type,
                k.content,
                s.distance AS similarity_score,
                k.source_url
            FROM search_results s
            JOIN {self.NVIDIA_KNOWLEDGE_TABLE} k
              ON s.doc_id = k.doc_id
            ORDER BY s.distance ASC
        """

        # Query for Stack Overflow questions
        so_query = f"""
            WITH search_results AS (
                SELECT
                    base.doc_id,
                    distance
                FROM VECTOR_SEARCH(
                    TABLE {self.SO_EMBEDDINGS_TABLE},
                    'embedding',
                    (SELECT [{embedding_str}] AS query_vector),
                    top_k => {top_k},
                    distance_type => 'COSINE'
                )
            )
            SELECT
                'Stack Overflow' AS source_type,
                CONCAT('**Question:** ', k.title, '\\n\\n**Answer:** ', k.answer) AS content,
                s.distance AS similarity_score,
                k.source_url
            FROM search_results s
            JOIN {self.SO_KNOWLEDGE_TABLE} k
              ON s.doc_id = k.doc_id
            ORDER BY s.distance ASC
        """


        # Run queries in parallel
        nvidia_job = self.client.query(nvidia_query)
        so_job = self.client.query(so_query)

        # Combine and sort results
        all_results = list(nvidia_job.result()) + list(so_job.result())
        all_results.sort(key=lambda x: x.similarity_score)

        return all_results[:top_k]

    def _display_search_metrics(self, docs):
        table = Table(title="📊 Search & Retrieval Metrics", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="bold blue", width=15)
        table.add_column("Document Snippet", style="cyan", no_wrap=True, width=70)
        table.add_column("Source URL", style="green", no_wrap=True, width=50)
        table.add_column("Confidence", justify="right", style="bold yellow")

        for i, doc in enumerate(docs, 1):
            confidence = (1 - doc.similarity_score) * 100
            snippet = doc.content.replace('\n', ' ').strip()

            confidence_text = f"{confidence:.1f}%"
            if confidence > 75:
                color = "green"
            elif confidence > 50:
                color = "yellow"
            else:
                color = "red"

            table.add_row(
                str(i),
                doc.source_type,
                snippet[:68] + "..." if len(snippet) > 70 else snippet,
                doc.source_url,
                Text(confidence_text, style=color)
            )
        self.console.print(table)

    def _display_performance_metrics(self, timings):
        table = Table(title="⏱️ Performance Metrics", show_header=True, header_style="bold blue")
        table.add_column("Stage", style="cyan")
        table.add_column("Duration (s)", style="magenta", justify="right")
        table.add_column("Percentage", style="green", justify="right")

        total_time = sum(timings.values())
        for stage, duration in timings.items():
            percentage = (duration / total_time * 100) if total_time > 0 else 0
            table.add_row(stage, f"{duration:.3f}", f"{percentage:.1f}%")

        table.add_section()
        table.add_row("Total", f"{total_time:.3f}", "100.0%")
        self.console.print(table)

    def display_cosine_wave_similarity(self, docs):
        """
        Visualizes similarity scores as a series of cosine waves.
        The amplitude and color of each wave are determined by the similarity score.
        """
        if not docs:
            return


        fig = go.Figure()

        # Prepare data for plotting
        scores = [(1 - doc.similarity_score) for doc in docs]
        titles = [doc.content.split('\n')[0][:50] + '...' for doc in docs]
        num_docs = len(scores)

        # Create a continuous x-axis for all waves
        x_vals = np.linspace(0, 2 * np.pi * num_docs, 100 * num_docs)

        for i, (score, title) in enumerate(zip(scores, titles)):
            # Each wave gets its own 2*pi segment on the x-axis
            x_segment = np.linspace(i * 2 * np.pi, (i + 1) * 2 * np.pi, 100)
            y_vals = score * np.cos(x_segment - i * 2 * np.pi) # Amplitude is the score

            fig.add_trace(go.Scatter(
                x=x_segment,
                y=y_vals,
                mode='lines',
                name=f'{score:.2f}: {title}',
                line=dict(color=px.colors.sequential.Viridis[int(score * (len(px.colors.sequential.Viridis) -1))]),
                hoverinfo='name'
            ))

            # Add an annotation for each document at the peak of its wave
            fig.add_annotation(
                x=i * 2 * np.pi,
                y=score,
                text=f"Doc {i+1}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )

        fig.update_layout(
            title='Cosine Similarity Waveform',
            xaxis_title='Documents',
            yaxis_title='Similarity Score (Amplitude)',
            yaxis=dict(range=[-1.1, 1.1]),
            xaxis=dict(
                tickmode='array',
                tickvals=[i * 2 * np.pi for i in range(num_docs)],
                ticktext=[f'Doc {i+1}' for i in range(num_docs)]
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        fig.show(renderer="notebook")

    def generate_answer(self, question):
        """Generate conversational answer using ONLY BigQuery embeddings context"""

        timings = {}

        # --- 1. Vector Search ---
        start_time = time.time()
        similar_docs = self.search_similar_documents(question, top_k=3)
        timings['Vector Search'] = time.time() - start_time

        if not similar_docs:
            self.console.print("[bold red]I couldn't find relevant information in our NVIDIA documentation.[/bold red]")
            return

        self._display_search_metrics(similar_docs)
        self.display_cosine_wave_similarity(similar_docs)

        # --- 2. Build context from search results ---
        start_time = time.time()
        context_parts = [f"source: {doc.source_url}\ncontent: {doc.content}" for doc in similar_docs]
        sources = [f"source: {doc.source_url}\n" for doc in similar_docs]
        timings['Content Fetching'] = time.time() - start_time
        context_text = "\n\n---\n\n".join(context_parts)
        sources_text = "\n\n---\n\n".join(sources)
        self.console.print(Panel(sources_text, title="[bold blue]📝 Context for LLM[/bold blue]", expand=False))

        prompt = f"""
        **Role**: You are an enthusiastic NVIDIA GPU expert assistant. You love helping developers with CUDA, GPU programming, and AI technologies.

        **Context from NVIDIA Documentation**:
        {context_text}

        **User Question**: {question}

        **Instructions**:
        - Answer conversationally and helpfully, like a knowledgeable colleague.
        - Use bullet points or numbered steps when explaining complex topics.
        - Show enthusiasm for NVIDIA technologies.
        - Keep it professional but friendly.
        - Use emojis sparingly to make it engaging.
        - Always base your answer strictly on the context provided.

        **Your Response**:
        """

        # --- 3. Answer Generation ---
        start_time = time.time()
        response = self.gen_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1024,
                "top_p": 0.9
            }
        )
        timings['Answer Generation'] = time.time() - start_time

        self.console.print(Panel(Markdown(response.text), title="[bold green]💡 NVIDIA AI Assistant Says...[/bold green]"))
        self._display_performance_metrics(timings)


def run_assistant():
    expert = NVIDIAExpertSystem()
    expert.console.print(Panel("[[bold green]🚀 NVIDIA AI Assistant Initialized[/bold green]]", title="✅ System Ready", expand=False))

    questions = [
        "What is CUDA memory coalescing and why is it important?",
        "How can I optimize CUDA kernels for better performance?",
        "What are the differences between shared memory and global memory in CUDA?",
        "Explain the concept of warp divergence in CUDA.",
        "How do CUDA streams help with concurrency?",
    ]

    # --- UI Components ---
    question_buttons = [widgets.Button(description=q, layout=widgets.Layout(width='95%')) for q in questions]
    custom_question_text = widgets.Text(placeholder='Or type your own question here...', layout=widgets.Layout(width='70%'))
    custom_question_button = widgets.Button(description="Ask Assistant", button_style='success')
    output_area = widgets.Output()

    def ask_question(question_text):
        with output_area:
            clear_output()
            expert.console.print(Panel(f"[bold yellow]❓ Asking[/bold yellow]: {question_text}", title="User Question"))
            expert.generate_answer(question_text)

    def on_button_clicked(b):
        ask_question(b.description)

    def on_custom_button_clicked(b):
        if custom_question_text.value:
            ask_question(custom_question_text.value)

    for btn in question_buttons:
        btn.on_click(on_button_clicked)
    custom_question_button.on_click(on_custom_button_clicked)

    # --- Layout ---
    expert.console.print(Panel("[bold cyan]Select a question or enter your own below:[/bold cyan]"))
    buttons_box = widgets.VBox(question_buttons)
    custom_input_box = widgets.HBox([custom_question_text, custom_question_button])
    display(buttons_box, custom_input_box, output_area)

if __name__ == "__main__":
    run_assistant()