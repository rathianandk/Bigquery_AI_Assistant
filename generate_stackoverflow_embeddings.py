import os
import logging
import time
import vertexai
from google.cloud import bigquery
from vertexai.language_models import TextEmbeddingModel
from tqdm import tqdm

# --- Configuration ---
LOGGING_LEVEL = logging.INFO
PROJECT_ID = os.environ.get('GOOGLE_CLOUD_PROJECT', 'precise-mystery-466919-u5')
LOCATION = "us-central1"

# Source table (clone in us-central1)
SOURCE_TABLE = "`precise-mystery-466919-u5.nvidia_docs_qa.stackoverflow_knowledge_clone`"

# Destination table for embeddings
destination_table_id = "precise-mystery-466919-u5.nvidia_docs_qa.stackoverflow_embeddings"

VERTEX_MODEL = 'text-embedding-005' # Use a model that produces 768d embeddings
FETCH_BATCH_SIZE = 500 # How many rows to fetch from BQ at a time
EMBEDDING_BATCH_SIZE = 5 # Vertex AI API has a limit of 5 per request

# Setup logging
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embeddings_in_batches(model, texts):
    """Generates embeddings for a list of texts in batches."""
    all_embeddings = []
    for i in tqdm(range(0, len(texts), EMBEDDING_BATCH_SIZE), desc="      Generating Embeddings"):
        batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
        try:
            embeddings = model.get_embeddings(batch_texts)
            all_embeddings.extend([list(e.values) for e in embeddings])
        except Exception as e:
            logging.error(f"Error getting embeddings for batch {i//EMBEDDING_BATCH_SIZE}: {e}")
            all_embeddings.extend([[]] * len(batch_texts))
    return all_embeddings

def generate_embeddings():
    """
    Reads data from a BigQuery table, generates embeddings using Vertex AI in batches,
    and inserts them into another BigQuery table.
    """
    logging.info("🚀 Starting Stack Overflow embedding generation process...")

    # --- 1. Initialize Vertex AI & BigQuery --- 
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    model = TextEmbeddingModel.from_pretrained(VERTEX_MODEL)
    client = bigquery.Client(project=PROJECT_ID)
    logging.info(f"✅ Vertex AI and BigQuery clients initialized in {LOCATION}.")
    logging.info("Waiting 10 seconds for BigQuery table to be available for streaming...")
    time.sleep(10)

    # --- 2. Fetch total row count for progress bar ---
    query = f"SELECT COUNT(*) as total_rows FROM {SOURCE_TABLE}"
    total_rows = client.query(query).to_dataframe()['total_rows'][0]
    logging.info(f"Found {total_rows} documents to process.")

    # --- 3. Process data in batches ---
    processed_rows = 0
    with tqdm(total=total_rows, desc="Overall Progress") as pbar:
        for offset in range(0, total_rows, FETCH_BATCH_SIZE):
            # Fetch a batch of documents from BigQuery
            query = f"""
                SELECT doc_id, title, question, answer, tags, source_url
                FROM {SOURCE_TABLE}
                ORDER BY doc_id
                LIMIT {FETCH_BATCH_SIZE} OFFSET {offset}
            """
            logging.info(f"\nFetching rows {offset} to {offset + FETCH_BATCH_SIZE}...")
            df = client.query(query).to_dataframe()

            if df.empty:
                break

            # Prepare content for embedding
            df['content'] = df['title'].fillna('') + ' ' + \
                            df['question'].fillna('') + ' ' + \
                            df['source_url'].fillna('') + ' ' + \
                            df['answer'].fillna('') + ' ' + \
                            df['tags'].fillna('')
            
            texts_to_embed = df['content'].tolist()
            doc_ids = df['doc_id'].tolist()

            # Generate embeddings
            embeddings = get_embeddings_in_batches(model, texts_to_embed)

            # Prepare data for insertion
            rows_to_insert = []
            for i, emb in enumerate(embeddings):
                if emb: # Only insert if embedding was successful
                    rows_to_insert.append({
                        'doc_id': doc_ids[i],
                        'embedding': emb
                    })
            
            # Insert embeddings into BigQuery
            if rows_to_insert:
                logging.info(f"Inserting {len(rows_to_insert)} embeddings into BigQuery...")
                errors = client.insert_rows_json(destination_table_id, rows_to_insert)
                if errors:
                    logging.error(f"Encountered errors while inserting rows: {errors}")
                else:
                    logging.info("✅ Batch inserted successfully.")
            
            processed_rows += len(df)
            pbar.update(len(df))

    logging.info("🎉 Successfully generated and inserted all embeddings!")

if __name__ == "__main__":
    generate_embeddings()
