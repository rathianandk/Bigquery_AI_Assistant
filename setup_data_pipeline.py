#!/usr/bin/env python3
"""
BigQuery-Native AI Developer Assistant - Data Pipeline Setup
Ingests Stack Overflow, NVIDIA forums, and internal docs into BigQuery
Uses only BigQuery-native AI capabilities
"""

import os
import json
import pandas as pd
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import requests
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BigQueryNativeAISetup:
    """Setup BigQuery-native AI developer assistant data pipeline"""
    
    def __init__(self, project_id: str = "precise-mystery-466919-u5"):
        self.project_id = project_id or os.environ.get('GOOGLE_CLOUD_PROJECT')
        if not self.project_id:
            raise ValueError("Please set GOOGLE_CLOUD_PROJECT environment variable")
        
        self.client = bigquery.Client(project=self.project_id)
        self.dataset_id = "developer_assistant"
        self.dataset_ref = f"{self.project_id}.{self.dataset_id}"
        
    def create_dataset_and_tables(self):
        """Create BigQuery dataset and tables for the AI assistant"""
        logger.info("Creating BigQuery dataset and tables...")
        
        # Create dataset
        try:
            dataset = bigquery.Dataset(f"{self.project_id}.{self.dataset_id}")
            dataset.location = "US"
            dataset.description = "BigQuery-Native AI Developer Assistant Knowledge Base"
            self.client.create_dataset(dataset, exists_ok=True)
            logger.info(f"Dataset {self.dataset_id} created/verified")
        except Exception as e:
            logger.error(f"Error creating dataset: {e}")
            raise
        
        # SQL to create all tables
        create_tables_sql = f"""
        -- Stack Overflow posts table
        CREATE OR REPLACE TABLE `{self.dataset_ref}.stackoverflow_posts` (
            post_id INT64,
            post_type STRING,  -- 'question' or 'answer'
            title STRING,
            body STRING,
            tags ARRAY<STRING>,
            score INT64,
            view_count INT64,
            creation_date TIMESTAMP,
            owner_user_id INT64,
            accepted_answer_id INT64,
            parent_id INT64,  -- For answers, links to question
            source STRING DEFAULT 'stackoverflow'
        );
        
        -- NVIDIA forum posts table
        CREATE OR REPLACE TABLE `{self.dataset_ref}.nvidia_forum_posts` (
            post_id STRING,
            thread_id STRING,
            title STRING,
            content STRING,
            category STRING,
            tags ARRAY<STRING>,
            author STRING,
            creation_date TIMESTAMP,
            reply_count INT64,
            view_count INT64,
            is_solution BOOL,
            source STRING DEFAULT 'nvidia_forums'
        );
        
        -- Internal documentation table
        CREATE OR REPLACE TABLE `{self.dataset_ref}.internal_docs` (
            doc_id STRING,
            title STRING,
            content STRING,
            doc_type STRING,  -- 'brd', 'api_doc', 'troubleshooting', 'faq'
            category STRING,
            tags ARRAY<STRING>,
            last_updated TIMESTAMP,
            author STRING,
            version STRING,
            source STRING DEFAULT 'internal_docs'
        );
        
        -- Support tickets table
        CREATE OR REPLACE TABLE `{self.dataset_ref}.support_tickets` (
            ticket_id STRING,
            title STRING,
            description STRING,
            resolution STRING,
            category STRING,
            priority STRING,
            status STRING,
            created_date TIMESTAMP,
            resolved_date TIMESTAMP,
            tags ARRAY<STRING>,
            time_to_resolution_hours FLOAT64,
            source STRING DEFAULT 'support_tickets'
        );
        
        -- Unified knowledge index with embeddings
        CREATE OR REPLACE TABLE `{self.dataset_ref}.unified_knowledge_index` (
            knowledge_id STRING,
            source_table STRING,
            source_id STRING,
            title STRING,
            content_chunk STRING,
            chunk_index INT64,
            embedding ARRAY<FLOAT64>,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        );
        
        -- Analysis results cache
        CREATE OR REPLACE TABLE `{self.dataset_ref}.analysis_cache` (
            query_hash STRING,
            input_problem STRING,
            analysis_result JSON,
            sources_used ARRAY<STRING>,
            confidence_score FLOAT64,
            processing_time_seconds FLOAT64,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        );
        
        -- Usage analytics
        CREATE OR REPLACE TABLE `{self.dataset_ref}.usage_analytics` (
            session_id STRING,
            user_id STRING,
            query_text STRING,
            response_time_seconds FLOAT64,
            satisfaction_rating INT64,
            sources_count INT64,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        );
        """
        
        # Execute table creation
        try:
            query_job = self.client.query(create_tables_sql)
            query_job.result()
            logger.info("All tables created successfully")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def setup_ml_models(self):
        """Create BigQuery ML models for embeddings and text generation"""
        logger.info("Setting up BigQuery ML models...")
        
        ml_models_sql = f"""
        -- Text embedding model
        CREATE OR REPLACE MODEL `{self.dataset_ref}.text_embedding_model`
                        REMOTE WITH CONNECTION `projects/{self.project_id}/locations/us/connections/bq-connection-us`
        OPTIONS (
            ENDPOINT = 'text-embedding-005'
        );

        -- Text generation model
        CREATE OR REPLACE MODEL `{self.dataset_ref}.text_generation_model`
                        REMOTE WITH CONNECTION `projects/{self.project_id}/locations/us/connections/bq-connection-us`
                        OPTIONS (endpoint = 'gemini-2.5-flash-lite');

        -- Time series forecasting model (Commented out due to single-timestamp data)
        -- CREATE OR REPLACE MODEL `{self.dataset_ref}.forecasting_model`
        --                 OPTIONS(
        --                     MODEL_TYPE='ARIMA_PLUS',
        --                     TIME_SERIES_TIMESTAMP_COL='date',
        --                     TIME_SERIES_DATA_COL='issue_count',
        --                     TIME_SERIES_ID_COL='source_keyword'
        --                 )
        -- AS
        -- SELECT
        --     DATE(created_at) AS date,
        --     COUNT(*) AS issue_count,
        --     source_table AS source_keyword
        -- FROM
        --     `{self.dataset_ref}.unified_knowledge_index`
        -- GROUP BY
        --     date, source_keyword;

        -- Forecasting model for ticket volume prediction (commented out)
        -- CREATE OR REPLACE MODEL `{self.dataset_ref}.ticket_forecast_model`
        -- OPTIONS (
        --     model_type='ARIMA_PLUS',
        --     time_series_timestamp_col='date',
        --     time_series_data_col='ticket_count'
        -- ) AS
        -- SELECT
        --   DATE(created_date) AS date,
        --   COUNT(ticket_id) AS ticket_count
        -- FROM
        --   `{self.dataset_ref}.support_tickets`
        -- GROUP BY 1;
        """
        
        try:
            query_job = self.client.query(ml_models_sql)
            query_job.result()
            logger.info("ML models created successfully")
        except Exception as e:
            logger.error(f"Error creating ML models: {e}")
            raise
    
    def ingest_stackoverflow_data(self):
        """Ingest high-quality Stack Overflow Q&A pairs"""
        logger.info("Ingesting Stack Overflow Q&A pairs...")
        
        # Enhanced Q&A extraction with your query pattern
        stackoverflow_qa_sql = f"""
        -- Create high-quality Q&A pairs table
        CREATE OR REPLACE TABLE `{self.dataset_ref}.stackoverflow_qa_pairs` AS
        WITH quality_qa_pairs AS (
          SELECT
            q.id as question_id,
            a.id as answer_id,
            q.title,
                                    CONCAT(q.title, '\n\n', q.body) AS question_text,
            a.body AS answer_text,
            q.tags,
            q.score as question_score,
            a.score as answer_score,
            q.view_count,
            q.creation_date as question_date,
            a.creation_date as answer_date,
            -- Quality indicators
            LENGTH(q.body) as question_length,
            LENGTH(a.body) as answer_length,
            -- Extract key technology mentions
            CASE 
              WHEN LOWER(q.tags) LIKE '%cuda%' OR LOWER(q.title) LIKE '%cuda%' THEN 'CUDA'
              WHEN LOWER(q.tags) LIKE '%tensorrt%' OR LOWER(q.title) LIKE '%tensorrt%' THEN 'TensorRT'
              WHEN LOWER(q.tags) LIKE '%pytorch%' OR LOWER(q.title) LIKE '%pytorch%' THEN 'PyTorch'
              WHEN LOWER(q.tags) LIKE '%tensorflow%' OR LOWER(q.title) LIKE '%tensorflow%' THEN 'TensorFlow'
              WHEN LOWER(q.tags) LIKE '%nvidia%' OR LOWER(q.title) LIKE '%nvidia%' THEN 'NVIDIA'
              ELSE 'GPU'
            END as primary_category,
            -- Error code extraction
            REGEXP_EXTRACT(CONCAT(q.title, ' ', q.body), r'(?i)(error\s+\d+|cuda\s+error\s+\d+)') as error_code
          FROM
            `bigquery-public-data.stackoverflow.posts_questions` q
          JOIN
            `bigquery-public-data.stackoverflow.posts_answers` a
          ON
            q.accepted_answer_id = a.id
          WHERE
            q.accepted_answer_id IS NOT NULL
            AND a.body IS NOT NULL
            AND LENGTH(a.body) > 50
            AND LENGTH(q.body) > 20
            AND (
              q.tags LIKE '%nvidia%' OR 
              q.tags LIKE '%cuda%' OR 
              q.tags LIKE '%gpu%' OR
              q.tags LIKE '%tensorrt%' OR
              q.tags LIKE '%pytorch%' OR
              q.tags LIKE '%tensorflow%' OR
              LOWER(q.title) LIKE '%nvidia%' OR
              LOWER(q.title) LIKE '%cuda%' OR
              LOWER(q.title) LIKE '%gpu%'
            )
            AND q.score >= 0
            AND a.score >= 1
            AND q.creation_date >= '2018-01-01'
        )
        SELECT *
        FROM quality_qa_pairs
        WHERE 
          question_length BETWEEN 50 AND 5000
          AND answer_length BETWEEN 50 AND 8000
        ORDER BY 
          question_score DESC, 
          answer_score DESC,
          view_count DESC
        LIMIT 5000;
        
        -- Also populate the original posts table for compatibility
        INSERT INTO `{self.dataset_ref}.stackoverflow_posts`
        SELECT 
            question_id as post_id,
            'question' as post_type,
            title,
            question_text as body,
            SPLIT(tags, '|') as tags,
            question_score as score,
            view_count,
            question_date as creation_date,
            NULL as owner_user_id,
            answer_id as accepted_answer_id,
            NULL as parent_id,
            'stackoverflow' as source
        FROM `{self.dataset_ref}.stackoverflow_qa_pairs`
        LIMIT 1000;
        """
        
        try:
            query_job = self.client.query(stackoverflow_qa_sql)
            query_job.result()
            logger.info("Stack Overflow Q&A pairs ingested successfully")
        except Exception as e:
            logger.error(f"Error ingesting Stack Overflow data: {e}")
            raise
    
    def ingest_simulated_nvidia_forums(self):
        """Ingest simulated NVIDIA forum data"""
        logger.info("Ingesting simulated NVIDIA forum data...")
        
        # Simulated NVIDIA forum posts
        nvidia_posts = [
            {
                "post_id": "nv_001",
                "thread_id": "thread_001", 
                "title": "CUDA Error 702: Launch timeout on RTX 4090",
                "content": "Getting CUDA error 702 after updating to driver 536.25. The error occurs during large tensor operations. Workstation has RTX 4090 with 24GB VRAM. Error message: 'CUDA error 702: launch timeout or launch failure'. This started happening after the latest driver update.",
                "category": "CUDA Programming",
                "tags": ["cuda", "rtx4090", "error702", "driver"],
                "author": "dev_user_1",
                "creation_date": "2024-01-15 10:30:00",
                "reply_count": 8,
                "view_count": 1250,
                "is_solution": False
            },
            {
                "post_id": "nv_002", 
                "thread_id": "thread_001",
                "title": "Re: CUDA Error 702: Launch timeout on RTX 4090",
                "content": "This is a known issue with driver 536.25. The timeout occurs because the kernel execution exceeds the TDR (Timeout Detection and Recovery) limit. Solution: 1) Downgrade to driver 531.79, 2) Increase TDR timeout in registry, or 3) Split large operations into smaller chunks. I recommend option 1 for immediate fix.",
                "category": "CUDA Programming", 
                "tags": ["cuda", "rtx4090", "error702", "solution"],
                "author": "nvidia_expert",
                "creation_date": "2024-01-15 14:20:00",
                "reply_count": 0,
                "view_count": 1250,
                "is_solution": True
            },
            {
                "post_id": "nv_003",
                "thread_id": "thread_002",
                "title": "TensorRT optimization fails on A100",
                "content": "TensorRT engine build failing on A100 with error 'Invalid argument'. Model works fine on V100. Using TensorRT 8.6.1 with CUDA 12.1. The error occurs during the optimization phase when building the engine from ONNX model.",
                "category": "TensorRT",
                "tags": ["tensorrt", "a100", "optimization", "onnx"],
                "author": "ml_engineer",
                "creation_date": "2024-01-20 09:15:00", 
                "reply_count": 5,
                "view_count": 890,
                "is_solution": False
            }
        ]
        
        # Convert to DataFrame and insert
        df = pd.DataFrame(nvidia_posts)
        df['creation_date'] = pd.to_datetime(df['creation_date'])
        
        # Insert into BigQuery
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = self.client.load_table_from_dataframe(
            df, f"{self.dataset_ref}.nvidia_forum_posts", job_config=job_config
        )
        job.result()
        logger.info("NVIDIA forum data ingested successfully")
    
    def ingest_internal_docs(self):
        """Ingest simulated internal documentation"""
        logger.info("Ingesting internal documentation...")
        
        internal_docs = [
            {
                "doc_id": "brd_001",
                "title": "GPU Computing Best Practices",
                "content": "This document outlines best practices for GPU computing in our development environment. Key guidelines: 1) Always check CUDA compatibility before driver updates, 2) Use TDR timeout adjustments for long-running kernels, 3) Implement proper error handling for CUDA operations, 4) Monitor GPU memory usage to prevent OOM errors.",
                "doc_type": "brd",
                "category": "Development Guidelines",
                "tags": ["gpu", "cuda", "best-practices"],
                "last_updated": "2024-01-10 12:00:00",
                "author": "tech_lead",
                "version": "2.1"
            },
            {
                "doc_id": "api_001", 
                "title": "CUDA Error Handling API Reference",
                "content": "Complete reference for CUDA error codes and handling. Error 702 (cudaErrorLaunchTimeout): Occurs when kernel execution exceeds TDR limit. Common causes: infinite loops, excessive computation, driver issues. Solutions: reduce kernel complexity, increase TDR timeout, update drivers.",
                "doc_type": "api_doc",
                "category": "API Reference",
                "tags": ["cuda", "error-handling", "api"],
                "last_updated": "2024-01-05 15:30:00",
                "author": "api_team",
                "version": "1.5"
            }
        ]
        
        df = pd.DataFrame(internal_docs)
        df['last_updated'] = pd.to_datetime(df['last_updated'])
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        job = self.client.load_table_from_dataframe(
            df, f"{self.dataset_ref}.internal_docs", job_config=job_config
        )
        job.result()
        logger.info("Internal documentation ingested successfully")
    
    def generate_embeddings(self):
        """Generate embeddings for all content using BigQuery ML"""
        logger.info("Generating embeddings for unified knowledge index...")
        
        embeddings_sql = f"""
        -- Clear existing embeddings
        DELETE FROM `{self.dataset_ref}.unified_knowledge_index` WHERE TRUE;
        
        -- Generate embeddings for Stack Overflow posts
        INSERT INTO `{self.dataset_ref}.unified_knowledge_index`
        SELECT
            CONCAT('so_', CAST(t.post_id AS STRING)) as knowledge_id,
            'stackoverflow_posts' as source_table,
            CAST(t.post_id AS STRING) as source_id,
            COALESCE(t.title, 'Stack Overflow Post') as title,
            e.content as content_chunk,
            1 as chunk_index,
            e.ml_generate_embedding_result as embedding,
            TO_JSON(STRUCT(
                t.post_type,
                t.tags,
                t.score,
                t.creation_date
            )) as metadata,
            CURRENT_TIMESTAMP() as created_at
        FROM
            ML.GENERATE_EMBEDDING(
                MODEL `{self.dataset_ref}.text_embedding_model`,
                (SELECT post_id, CONCAT(COALESCE(title, ''), ' ', SUBSTR(body, 1, 1000)) as content FROM `{self.dataset_ref}.stackoverflow_posts` WHERE body IS NOT NULL LIMIT 1000)
            ) AS e
        JOIN
            `{self.dataset_ref}.stackoverflow_posts` AS t
        ON
            t.post_id = e.post_id;
        
        -- Generate embeddings for NVIDIA forum posts  
        INSERT INTO `{self.dataset_ref}.unified_knowledge_index`
        SELECT
            CONCAT('nv_', t.post_id) as knowledge_id,
            'nvidia_forum_posts' as source_table,
            t.post_id as source_id,
            t.title,
            e.content as content_chunk,
            1 as chunk_index,
            e.ml_generate_embedding_result as embedding,
            TO_JSON(STRUCT(
                t.category,
                t.tags,
                t.is_solution,
                t.creation_date
            )) as metadata,
            CURRENT_TIMESTAMP() as created_at
        FROM
            ML.GENERATE_EMBEDDING(
                MODEL `{self.dataset_ref}.text_embedding_model`,
                (SELECT post_id, CONCAT(title, ' ', content) as content FROM `{self.dataset_ref}.nvidia_forum_posts`)
            ) AS e
        JOIN
            `{self.dataset_ref}.nvidia_forum_posts` AS t
        ON
            t.post_id = e.post_id;
        
        -- Generate embeddings for internal docs
        INSERT INTO `{self.dataset_ref}.unified_knowledge_index`
        SELECT
            CONCAT('doc_', t.doc_id) as knowledge_id,
            'internal_docs' as source_table,
            t.doc_id as source_id,
            t.title,
            e.content as content_chunk,
            1 as chunk_index,
            e.ml_generate_embedding_result as embedding,
            TO_JSON(STRUCT(
                t.doc_type,
                t.category,
                t.tags,
                t.last_updated
            )) as metadata,
            CURRENT_TIMESTAMP() as created_at
        FROM
            ML.GENERATE_EMBEDDING(
                MODEL `{self.dataset_ref}.text_embedding_model`,
                (SELECT doc_id, CONCAT(title, ' ', content) as content FROM `{self.dataset_ref}.internal_docs`)
            ) AS e
        JOIN
            `{self.dataset_ref}.internal_docs` AS t
        ON
            t.doc_id = e.doc_id;
        """
        
        try:
            query_job = self.client.query(embeddings_sql)
            query_job.result()
            logger.info("Embeddings generated successfully")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    def ingest_from_sources(self):
        """Ingest data from all configured sources and generate embeddings."""
        logger.info("Ingesting data from all sources...")
        self.ingest_stackoverflow_data()
        self.ingest_simulated_nvidia_forums()
        self.ingest_internal_docs()
        # After ingesting, generate embeddings
        self.generate_embeddings()

    def create_vector_index(self):
        """Create vector index for fast similarity search"""
        logger.info("Creating vector index...")
        
        vector_index_sql = f"""
        CREATE OR REPLACE VECTOR INDEX knowledge_vector_index
        ON `{self.dataset_ref}.unified_knowledge_index`(embedding)
        OPTIONS (
            index_type = 'IVF',
            distance_type = 'COSINE'
        );
        """
        
        try:
            query_job = self.client.query(vector_index_sql)
            query_job.result()
            logger.info("Vector index created successfully")
        except Exception as e:
            logger.warning(f"Vector index creation failed (may not be available in all regions): {e}")
    
    def run_setup(self):
        """Run complete setup pipeline"""
        logger.info("Starting BigQuery-Native AI Assistant setup...")
        
        try:
            # Core setup
            self.create_dataset_and_tables()

            # Data ingestion and embedding generation
            self.ingest_from_sources()

            # ML Model setup (after data is available)
            self.setup_ml_models()
            # self.create_vector_index() # Commented out for now
            
            logger.info("✅ Setup completed successfully!")
            logger.info(f"Dataset: {self.dataset_ref}")
            logger.info("Ready to run AI assistant queries!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise

def main():
    """Main setup function"""
    setup = BigQueryNativeAISetup()
    setup.run_setup()

if __name__ == "__main__":
    main()
