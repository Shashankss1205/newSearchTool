import os
import json
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import requests
from pinecone import Pinecone, ServerlessSpec
from fuzzywuzzy import fuzz
import re
from typing import List, Dict, Any, Optional
import ast
from dataclasses import dataclass, asdict
import logging
from dotenv import load_dotenv
import io
import time
import google.generativeai as generative_ai
import speech_recognition as sr
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import tempfile
import pickle
from sarvamai import SarvamAI
# import sqlite3
from datetime import datetime
from supabase import create_client, Client
from google import genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# class VoiceHandler:
#     """Handle voice input/output using Gemini"""
    
#     def __init__(self):
#         self.gemini_api_key = os.getenv('GOOGLE_API_KEY')
#         self.model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        
#         if self.gemini_api_key:
#             genai.configure(api_key=self.gemini_api_key)
#             self.model = genai.GenerativeModel(self.model_name)
#             logger.info("Gemini voice handler initialized")
#         else:
#             logger.warning("No Gemini API key found - voice features disabled")
#             self.model = None
    
#     def speech_to_text(self, audio_file) -> str:
#         """Convert speech to text"""
#         try:
#             r = sr.Recognizer()
#             with sr.AudioFile(audio_file) as source:
#                 audio = r.record(source)
#             return get_font(r.recognize_google(audio))
#         except Exception as e:
#             logger.error(f"Speech recognition error: {e}")
#             raise
    
#     def enhance_query_with_gemini(self, voice_query: str) -> str:
#         """Use Gemini to enhance voice query for better search"""
#         if not self.model:
#             return voice_query
            
#         try:
#             prompt = f"""
#             Convert this voice query into a better search query for a story search engine.
#             Focus on extracting themes, emotions, characters, settings, and events.
            
#             Voice query: "{voice_query}"
            
#             Return only the improved search query, nothing else.
#             """
            
#             response = self.model.generate_content(prompt)
#             return response.text.strip()
            
#         except Exception as e:
#             logger.error(f"Gemini enhancement error: {e}")
#             return voice_query
    
# import re

# def refine_semantic_query(query: str) -> str:
#     """
#     Cleans a user search query by removing story-related conversational phrases and generic keywords.

#     Args:
#         query (str): Raw user input.

#     Returns:
#         str: Refined search query.
#     """
#     query = query.strip().lower()

#     # Conversational patterns
#     phrase_patterns = [
#         r'^(tell|show|give|find|bring|recommend|suggest)\s+(me\s+)?(a|some)?\s*(story|stories|tale|tales|book|books|novel|novels)?\s*(of|about|on)?\s*',
#         r'^(a|some)?\s*(story|stories|tale|tales|book|books|novel|novels)?\s*(of|about|on)?\s*',
#         r'^(about|on|of)\s+'
#     ]

#     # Remove prefix phrases
#     for pattern in phrase_patterns:
#         query = re.sub(pattern, '', query)

#     # Remove generic words anywhere
#     generic_words = ['story', 'stories', 'tale', 'tales', 'book', 'books', 'novel', 'novels']
#     word_pattern = r'\b(?:' + '|'.join(generic_words) + r')\b'
#     query = re.sub(word_pattern, '', query)

#     # Cleanup extra whitespace
#     query = re.sub(r'\s+', ' ', query).strip()

#     return query

def decompose_query(query: str) -> List[str]:
    """Decompose a complex query into sub-queries using Gemini"""
    try:
        prompt = f"""
        Break down this search query into individual concepts/aspects that should be searched separately. 
        Return each concept on a new line, no numbering or bullets.
        If the query is simple (single concept), return just that concept.
        
        Examples:
        "motivational student" → 
        motivational
        student
        
        "success from Kerala" →
        success
        Kerala
        
        "friendship underwater animals" →
        friendship
        underwater animals

        "black cat" →
        black cat
        
        "good behaviour with siblings" →
        good behaviour with siblings

        Query to decompose: {query}
        """
        
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Split response into individual concepts
        concepts = [concept.strip() for concept in response.text.strip().split('\n') if concept.strip()]
        
        # Fallback to original query if decomposition fails
        return concepts if concepts else [query]
        
    except Exception as e:
        logger.error(f"Query decomposition error: {e}")
        return [query]  # Return original query as fallback

class VoiceHandler:
    """Handle speech-to-text using SarvamAI Saarika (v2.5)"""

    def __init__(self):
        self.api_key = os.getenv("SARVAM_API_KEY")
        if not self.api_key:
            raise RuntimeError("Please set SARVAM_API_KEY env var")
        self.client = SarvamAI(api_subscription_key=self.api_key)

    def speech_to_text(self, audio_path: str,
                       model: str = "saarika:v2.5",
                       language_code: str = "unknown"):
        """
        Transcribe audio via SarvamAI:
          • audio_path: path to .wav/.mp3 file
          • model: one of saarika:v1, v2, v2.5, flash
          • language_code: BCP‑47 code or 'unknown' for auto-detect
        """
        with open(audio_path, "rb") as f:
            resp = self.client.speech_to_text.transcribe(
                file=f,
                model=model,
                language_code=language_code,
            )
        # resp is a dict with keys: request_id, transcript, timestamps (if requested)
        print(resp)
        return resp.transcript
    
app = Flask(__name__)
CORS(app)
app.json_encoder = NumpyEncoder

class APIEmbeddingModel:
    """API-based embedding model supporting multiple providers"""
    
    def __init__(self):
        self.provider = os.getenv('EMBEDDING_PROVIDER', 'openai').lower()
        self.api_key = None
        self.base_url = None
        self.model_name = None
        self.dimension = None
        
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize the embedding provider"""
        if self.provider == 'openai':
            self.api_key = os.getenv('OPENAI_API_KEY')
            self.base_url = "https://api.openai.com/v1/embeddings"
            self.model_name = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
            self.dimension = 1536  # text-embedding-3-small dimension
            
        elif self.provider == 'google':
            self.api_key = os.getenv('GOOGLE_API_KEY')
            self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
            self.model_name = "embedding-001"
            self.dimension = 768
            
        elif self.provider == 'cohere':
            self.api_key = os.getenv('COHERE_API_KEY')
            self.base_url = "https://api.cohere.ai/v1/embed"
            self.model_name = os.getenv('COHERE_EMBEDDING_MODEL', 'embed-english-light-v3.0')
            self.dimension = 384
            
        elif self.provider == 'huggingface':
            self.api_key = os.getenv('HUGGINGFACE_API_KEY')
            self.model_name = os.getenv('HUGGINGFACE_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
            self.base_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_name}"
            self.dimension = 384
            
        else:
            # Fallback to local SentenceTransformer
            try:
                from sentence_transformers import SentenceTransformer
                self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.provider = 'local'
                self.dimension = 384
                logger.info("Using local SentenceTransformer model")
            except ImportError:
                raise ValueError("No valid embedding provider configured and sentence-transformers not available")
        
        if self.provider != 'local' and not self.api_key:
            logger.warning(f"No API key found for {self.provider}, falling back to local model")
            self._fallback_to_local()
        else:
            logger.info(f"Using {self.provider} embeddings with model: {self.model_name}")
    
    def _fallback_to_local(self):
        """Fallback to local model if API fails"""
        try:
            from sentence_transformers import SentenceTransformer
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.provider = 'local'
            self.dimension = 384
            logger.info("Fell back to local SentenceTransformer model")
        except ImportError:
            raise ValueError("Cannot fallback to local model - sentence-transformers not installed")
    
    def encode(self, texts: List[str], max_retries: int = 3) -> np.ndarray:
        """Encode texts to embeddings"""
        if self.provider == 'local':
            return self.local_model.encode(texts)
        
        for attempt in range(max_retries):
            try:
                if self.provider == 'openai':
                    return self._encode_openai(texts)
                elif self.provider == 'google':
                    return self._encode_google(texts)
                elif self.provider == 'cohere':
                    return self._encode_cohere(texts)
                elif self.provider == 'huggingface':
                    return self._encode_huggingface(texts)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.provider}: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All attempts failed, falling back to local model")
                    self._fallback_to_local()
                    return self.local_model.encode(texts)
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """OpenAI embeddings"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": texts,
            "model": self.model_name
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        embeddings = [item['embedding'] for item in result['data']]
        return np.array(embeddings)
    
    # def _encode_google(self, texts: List[str]) -> np.ndarray:
    #     """Google embeddings"""
    #     # embeddings = []
        
    #     # for text in texts:
    #     #     headers = {
    #     #         "Content-Type": "application/json"
    #     #     }
            
    #     #     data = {
    #     #         "model": f"models/{self.model_name}",
    #     #         "content": {
    #     #             "parts": [{"text": text}]
    #     #         }
    #     #     }
            
    #     #     url = f"{self.base_url}?key={self.api_key}"
    #     #     response = requests.post(url, headers=headers, json=data, timeout=30)
    #     #     response.raise_for_status()
            
    #     #     result = response.json()
    #     #     embeddings.append(result['embedding']['values'])
        
    #     # return np.array(embeddings)
    #     client = genai.Client()
    #     # result = [
    #     #     np.array(e.values) for e in client.models.embed_content(

    #     result = client.models.embed_content(
    #             model=self.model_name,
    #             contents=texts, 
    #             config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))

    #     return np.array(result.embeddings)
    def _encode_google(self, texts: List[str]) -> np.ndarray:
        """Google embeddings with validation and error handling"""
        client = genai.Client()

        try:
            result = client.models.embed_content(
                model=self.model_name,
                contents=texts,
                config=genai.types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
            )
        except Exception as e:
            # Network or API failure — let outer retry handle it
            raise RuntimeError(f"Google embedding request failed: {e}")

        # ✅ Validate response structure
        if not hasattr(result, "embeddings") or result.embeddings is None:
            raise ValueError("No embeddings returned from Google API")

        # Convert embeddings to the correct format
        try:
            embeddings = []
            for embedding in result.embeddings:
                # Check if it's already a numpy array
                if isinstance(embedding, np.ndarray):
                    embeddings.append(embedding)
                # If it has .values attribute (PaLM API format)
                elif hasattr(embedding, 'values'):
                    embeddings.append(np.array(embedding.values))
                else:
                    # Try converting directly to numpy array
                    embeddings.append(np.array(embedding))
        except Exception as e:
            raise ValueError(f"Failed to process embeddings: {e}")

        # ✅ Validate numeric values
        try:
            embeddings_array = np.array(embeddings, dtype=float)
        except (ValueError, TypeError):
            raise ValueError("Could not convert embeddings to numeric array")

        return np.array(embeddings, dtype=float)

    
    def _encode_cohere(self, texts: List[str]) -> np.ndarray:
        """Cohere embeddings"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "texts": texts,
            "model": self.model_name,
            "input_type": "search_document"
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return np.array(result['embeddings'])
    
    def _encode_huggingface(self, texts: List[str]) -> np.ndarray:
        """Hugging Face Inference API embeddings"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        
        response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return np.array(result)
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension

@dataclass
class Story:
    """Data class representing a story"""
    id: str
    filename: str
    character_primary: List[str]
    character_secondary: List[str]
    setting_primary: List[str]
    setting_secondary: List[str]
    theme_primary: List[str]
    theme_secondary: List[str]
    events_primary: List[str]
    events_secondary: List[str]
    emotions_primary: List[str]
    emotions_secondary: List[str]
    keywords: List[str]

@dataclass
class SearchResult:
    """Data class for search results"""
    story: Story
    score: float
    matched_fields: Dict[str, float]

COLUMNS_MAP = {
    "story_id": "filename",
    "story_title": None,
    "level": None,
    "success": None,
    "error": None,
    "validation_errors": None,
    "analysis_type": None,
    "characters_primary": "character_primary",
    "characters_secondary": "character_secondary",
    "settings_primary": "setting_primary",
    "settings_secondary": "setting_secondary",
    "themes_primary": "theme_primary",
    "themes_secondary": "theme_secondary",
    "themes_amazon": None,
    "events_primary": "events_primary",
    "events_secondary": "events_secondary",
    "emotions_primary": "emotions_primary",
    "emotions_secondary": "emotions_secondary",
    "keywords": "keywords",
    "processed_at": None
}

def format_column(df, COLUMNS_MAP=COLUMNS_MAP):
    
    df.rename(columns=COLUMNS_MAP, inplace=True) 
    df = df[list(val for val in COLUMNS_MAP.values() if val is not None)]
    return df

class StorySearchEngine:
    def __init__(self):
        self.model = APIEmbeddingModel()
        self.pc = None
        self.index = None
        self.stories = {}
        self.stories_backup_file = "stories_test_backup.pkl"
        
        # Initialize Pinecone
        self._init_pinecone()
        
        # Try to load existing stories from backup
        self._load_stories_backup()
        
        # If no stories loaded and Pinecone is available, try to sync from Pinecone
        # if not self.stories and self.index:
        #     self._sync_from_pinecone()

        # Field weights for scoring
        self.weights = {
            'theme_primary': 0.25,
            'theme_secondary': 0.15,
            'events_primary': 0.20,
            'events_secondary': 0.10,
            'emotions_primary': 0.15,
            'emotions_secondary': 0.08,
            'setting_primary': 0.12,
            'setting_secondary': 0.08,
            'character_primary': 0.15,
            'character_secondary': 0.10,
            'keywords': 0.12
        }
        
        # Semantic fields (will be embedded)
        self.semantic_fields = [
            'theme_primary', 'theme_secondary', 
            'events_primary', 'events_secondary',
            'emotions_primary', 'emotions_secondary',
            'setting_primary', 'setting_secondary'
        ]
        
        # Keyword fields (exact/fuzzy matching)
        self.keyword_fields = ['character_primary', 'character_secondary', 'keywords']
    
    def _init_pinecone(self):
        """Initialize Pinecone client and index"""
        try:
            api_key = os.getenv('PINECONE_API_KEY')
            if not api_key:
                logger.warning("PINECONE_API_KEY not found - running without Pinecone")
                return
            
            self.pc = Pinecone(api_key=api_key)
            
            # Use provider-specific index name to avoid dimension conflicts
            embedding_dim = self.model.get_sentence_embedding_dimension()
            index_name = f"story-search-{self.model.provider}-{embedding_dim}"
            test_db = os.getenv('TEST_DB')
            
            if test_db!='0':
                index_name = f"story-search-{self.model.provider}-{embedding_dim}-test-{test_db}"
            # Create index if it doesn't exist - use dynamic dimension
            if index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=index_name,
                    dimension=embedding_dim,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                logger.info(f"Created Pinecone index: {index_name} with dimension {embedding_dim}")
            else:
                # Verify existing index has correct dimension
                index_info = self.pc.describe_index(index_name)
                existing_dim = index_info.dimension
                if existing_dim != embedding_dim:
                    logger.warning(f"Existing index {index_name} has dimension {existing_dim}, expected {embedding_dim}")
                    # Delete and recreate index with correct dimension
                    logger.info(f"Deleting old index {index_name} and creating new one")
                    self.pc.delete_index(index_name)
                    time.sleep(5)  # Wait for deletion
                    self.pc.create_index(
                        name=index_name,
                        dimension=embedding_dim,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region='us-east-1'
                        )
                    )
                    logger.info(f"Created new Pinecone index: {index_name} with dimension {embedding_dim}")
            
            self.index = self.pc.Index(index_name)
            logger.info(f"Pinecone initialized successfully with index: {index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            self.pc = None
            self.index = None
    
    def _save_stories_supabase(self):
        """Save stories to Supabase storage"""
        
        FILE_PATH = self.stories_backup_file # local path
        bucket = os.getenv("SUPABASE_BUCKET")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_ANON_KEY")
        supabase = create_client(url, key)

        with open(FILE_PATH, "rb") as f:
            supabase.storage.from_(bucket).upload(FILE_PATH, f)
        # print(f"Upload status: {res.status_code} - {res.text}")
        logger.info(f"Uploaded stories to Supabase bucket {bucket} file {FILE_PATH}")   

    def _load_stories_supabase(self):
        url = os.getenv("SUPABASE_URL")
        anon_key = os.getenv("SUPABASE_ANON_KEY")
        bucket = os.getenv("SUPABASE_BUCKET")
        FILE_PATH = self.stories_backup_file # local path        
        # Initialize Supabase client
        supabase = create_client(url, anon_key)

        # Download file as bytes
        response = supabase.storage.from_(bucket).download(FILE_PATH)
        
        # Save the downloaded content locally
        with open(self.stories_backup_file, "wb") as f:
            f.write(response)
        logger.info(f"Loaded stories from Supabase bucket {bucket} file {FILE_PATH}")

    def _save_stories_backup(self):
        """Save stories to disk for persistence"""
        try:
            with open(self.stories_backup_file, 'wb') as f:
                pickle.dump(dict(self.stories), f)
            logger.info(f"Saved {len(self.stories)} stories to backup file")
            
            self._save_stories_supabase()
        except Exception as e:
            logger.error(f"Failed to save stories backup: {e}")
    
    def _load_stories_backup(self):
        """Load stories from disk backup"""
        try:
            self._load_stories_supabase()
            if os.path.exists(self.stories_backup_file):
                with open(self.stories_backup_file, 'rb') as f:
                    self.stories = pickle.load(f)

                logger.info(f"Loaded {len(self.stories)} stories from backup file")
            else:
                logger.info("No backup file found")
        except Exception as e:
            logger.error(f"Failed to load stories backup: {e}")
            self.stories = {}
    
    def _sync_from_pinecone(self):
        """Rebuild stories dict from Pinecone metadata"""
        if not self.index:
            logger.warning("Cannot sync from Pinecone - index not available")
            return
        
        try:
            logger.info("Attempting to sync stories from Pinecone metadata...")
            
            # Get index stats first
            stats = self.index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            logger.info(f"Found {total_vectors} vectors in Pinecone index")
            
            if total_vectors == 0:
                logger.info("No vectors found in Pinecone index")
                return
            
            # Query with a dummy vector to get all results
            dummy_vector = [0.0] * self.model.dimension
            
            # Fetch results in batches
            all_matches = []
            top_k = min(10000, total_vectors)  # Pinecone limit
            
            query_result = self.index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            all_matches.extend(query_result.get('matches', []))
            
            # Group by story_id to rebuild Story objects
            story_data = {}
            for match in all_matches:
                metadata = match.get('metadata', {})
                story_id = metadata.get('story_id')
                
                if not story_id:
                    continue
                
                if story_id not in story_data:
                    story_data[story_id] = {
                        'id': story_id,
                        'filename': metadata.get('filename', f"{story_id}.txt"),
                        'fields': {}
                    }
                
                field = metadata.get('field')
                text = metadata.get('text', '')
                
                if field and text:
                    # Convert text back to list (approximate)
                    field_values = [item.strip() for item in text.split(' ') if item.strip()]
                    story_data[story_id]['fields'][field] = field_values
            
            # Convert to Story objects
            synced_stories = {}
            for story_id, data in story_data.items():
                try:
                    story = Story(
                        id=story_id,
                        filename=data['filename'],
                        character_primary=data['fields'].get('character_primary', []),
                        character_secondary=data['fields'].get('character_secondary', []),
                        setting_primary=data['fields'].get('setting_primary', []),
                        setting_secondary=data['fields'].get('setting_secondary', []),
                        theme_primary=data['fields'].get('theme_primary', []),
                        theme_secondary=data['fields'].get('theme_secondary', []),
                        events_primary=data['fields'].get('events_primary', []),
                        events_secondary=data['fields'].get('events_secondary', []),
                        emotions_primary=data['fields'].get('emotions_primary', []),
                        emotions_secondary=data['fields'].get('emotions_secondary', []),
                        keywords=data['fields'].get('keywords', [])
                    )
                    synced_stories[story_id] = story
                except Exception as e:
                    logger.warning(f"Failed to create story object for {story_id}: {e}")
            
            if synced_stories:
                self.stories = synced_stories
                self._save_stories_backup()  # Save the synced data
                logger.info(f"Successfully synced {len(self.stories)} stories from Pinecone")
            else:
                logger.warning("No valid stories could be synced from Pinecone")
                
        except Exception as e:
            logger.error(f"Failed to sync from Pinecone: {e}")
    
    def _safe_eval_list(self, list_str: str) -> List[str]:
        """Safely evaluate string representation of list"""
        if pd.isna(list_str) or not list_str.strip():
            return []
        
        try:
            # Handle the case where it's already a list
            if isinstance(list_str, list):
                return list_str
             
            # Try to parse as JSON first
            if list_str.startswith('[') and list_str.endswith(']'):
                return ast.literal_eval(list_str)
            
            # Fallback: split by semi-colon
            return [item.strip().strip('"\'') for item in list_str.split(';')]
        except Exception as e:
            logger.warning(f"Failed to parse list string: {list_str}. Error: {e}")
            return []
    
    def load_data(self, csv_path: str = None, csv_data: str = None):
        """Load story data from CSV"""
        try:
            if csv_data:
                # Parse from string data
                df = pd.read_csv(io.StringIO(csv_data))
                if 'story_id' in df.columns:
                    df = format_column(df)
            elif csv_path:
                df = pd.read_csv(csv_path)
                df = df[~df.isin(['Error parsing','Error']).any(axis=1)]
                if 'story_id' in df.columns:
                    df = format_column(df)

            else:
                raise ValueError("Either csv_path or csv_data must be provided")
            
            logger.info(f"Loaded {len(df)} stories from CSV")
            
            # Convert to Story objects
            new_stories = {}
            for idx, row in df.iterrows():
                story_id = row['filename'].replace('.txt', '')
                # print(row['character_primary'])
                # print(self._safe_eval_list(row['character_primary']))
                story = Story(
                    id=story_id,
                    filename=row['filename'],
                    character_primary=self._safe_eval_list(row['character_primary']),
                    character_secondary=self._safe_eval_list(row['character_secondary']),
                    setting_primary=self._safe_eval_list(row['setting_primary']),
                    setting_secondary=self._safe_eval_list(row['setting_secondary']),
                    theme_primary=self._safe_eval_list(row['theme_primary']),
                    theme_secondary=self._safe_eval_list(row['theme_secondary']),
                    events_primary=self._safe_eval_list(row['events_primary']),
                    events_secondary=self._safe_eval_list(row['events_secondary']),
                    emotions_primary=self._safe_eval_list(row['emotions_primary']),
                    emotions_secondary=self._safe_eval_list(row['emotions_secondary']),
                    keywords=self._safe_eval_list(row['keywords'])
                )
                
                new_stories[story_id] = story
            # import json
            # from dataclasses import asdict

            # # Convert all Story dataclass instances to dictionaries
            # data_to_save = {story_id: asdict(story) for story_id, story in new_stories.items()}

            # # Save to JSON file
            # with open("stories.json", "w", encoding="utf-8") as f:
            #     json.dump(data_to_save, f, ensure_ascii=False, indent=2)

            # Update stories (don't replace, add to existing)
            self.stories.update(new_stories)
            logger.info(f"Total stories now: {len(self.stories)}")
            
            logger.info(f"Processed {len(self.stories)} stories")
            # Save backup
            # print('-'*50)
            # print(self.stories)
            # with open("temp.json", 'w', encoding='utf-8') as f:
            #     f.write(self.stories)
            # json_data = json.dumps(self.stories, default=lambda o: o.__dict__, indent=2)
            # with open("stories.json", "w", encoding="utf-8") as f:
            #     f.write(json_data)
            self._save_stories_backup()

            # Create embeddings and upsert to Pinecone
            if self.index:
                self._create_embeddings(new_stories)
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def _create_embeddings(self, stories_to_process: Dict[str, Story] = None):
        """Create embeddings for semantic fields and upsert to Pinecone"""
        try:
            if stories_to_process is None:
                stories_to_process = self.stories
            vectors_to_upsert = []
            
            # for story_id, story in stories_to_process.items():
            #     # Create combined text for each semantic field
            #     for field in self.semantic_fields:
            #         field_values = getattr(story, field)
            #         if field_values:
            #             combined_text = ' '.join(field_values)
                        
            #             # Create embedding
            #             embedding = self.model.encode([combined_text])[0].tolist()
                        
            #             # Create vector ID
            #             vector_id = f"{story_id}_{field}"
                        
            #             vectors_to_upsert.append({
            #                 'id': vector_id,
            #                 'values': embedding,
            #                 'metadata': {
            #                     'story_id': story_id,
            #                     'field': field,
            #                     'text': combined_text,
            #                     'filename': story.filename
            #                 }
            #             })
            # for story_id, story in stories_to_process.items():
            #     for field in self.semantic_fields:
            #         field_values = getattr(story, field)
            #         if field_values:
            #             # Create embedding
            #             embeddings = self.model.encode(field_values).tolist()
            #             # print(embeddings)
            #             counter = 0
            #             for embedding_obj,value in zip(embeddings,field_values):
            #                 embedding = embedding_obj.values
            #                 vector_id = f"{story_id}_{field}__{counter}"
            #                 counter += 1
            #                 vectors_to_upsert.append({
            #                     'id': vector_id,
            #                     'values': embedding,
            #                     'metadata': {
            #                         'story_id': story_id,
            #                         'field': field,
            #                         'text': value,
            #                         'filename': story.filename
            #                     }
            #                 })


            # ---------------------------------------------------------ONE VECTOR PER ITEM PER FIELD PER STORY---------------------------------------------------------

            story_counter=1
            for story_id, story in stories_to_process.items():
                logger.info(f"Processing story {story_id} ({story_counter}/{len(stories_to_process)})")
                story_counter += 1
                all_field_values = []
                for field in self.semantic_fields:
                    field_values = getattr(story, field)
                    if field_values:
                        for i, value in enumerate(field_values):
                            all_field_values.append([value, field])
                if not all_field_values:
                    logger.warning(f"No semantic fields found for story {story_id}, skipping")
                    continue
                # Create embedding
                embeddings = self.model.encode([sublist[0] for sublist in all_field_values]).tolist()
                # print(embeddings)
                counter = 0
                for embedding_obj,value_field_pair in zip(embeddings,all_field_values):
                    embedding = embedding_obj
                    vector_id = f"{story_id}_{value_field_pair[1]}__{counter}"
                    counter += 1
                    vectors_to_upsert.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': {
                            'story_id': story_id,
                            'field': value_field_pair[1],
                            'text': value_field_pair[0],
                            'filename': story.filename
                        }
                    })

            # ---------------------------------------------------------ONE VECTOR PER FIELD PER STORY---------------------------------------------------------

            # story_counter = 1
            # for story_id, story in stories_to_process.items():
            #     logger.info(f"Processing story {story_id} ({story_counter}/{len(stories_to_process)})")
            #     story_counter += 1

            #     field_texts = []
            #     for field in self.semantic_fields:
            #         field_values = getattr(story, field)
            #         if field_values:
            #             # Join all values in the field into one descriptive sentence
            #             field_sentence = f"{field.replace('_', ' ')} are: {', '.join(field_values)}"
            #             field_texts.append((field_sentence, field))

            #     if not field_texts:
            #         logger.warning(f"No semantic fields found for story {story_id}, skipping")
            #         continue

            #     # Create embeddings at the story level (one per field)
            #     embeddings = self.model.encode([ft[0] for ft in field_texts]).tolist()

            #     for idx, (embedding, (text, field)) in enumerate(zip(embeddings, field_texts)):
            #         vector_id = f"{story_id}_{field}__{idx}"
            #         vectors_to_upsert.append({
            #             'id': vector_id,
            #             'values': embedding,  # already a list from .tolist()
            #             'metadata': {
            #                 'story_id': story_id,
            #                 'field': field,
            #                 'text': text,
            #                 'filename': story.filename
            #             }
            #         })

            # ---------------------------------------------------------ONE VECTOR PER STORY---------------------------------------------------------
            # story_counter = 1
            # for story_id, story in stories_to_process.items():
            #     logger.info(f"Processing story {story_id} ({story_counter}/{len(stories_to_process)})")
            #     story_counter += 1

            #     all_field_texts = []
            #     for field in self.semantic_fields:
            #         field_values = getattr(story, field)
            #         if field_values:
            #             # Join all values for this field into a descriptive sentence
            #             field_sentence = f"{field.replace('_', ' ')} are: {', '.join(field_values)}"
            #             all_field_texts.append(field_sentence)

            #     if not all_field_texts:
            #         logger.warning(f"No semantic fields found for story {story_id}, skipping")
            #         continue

            #     # Combine all fields into one big descriptive text
            #     combined_text = " ".join(all_field_texts)

            #     # Create one embedding for the entire story
            #     embedding = self.model.encode([combined_text])[0].tolist()

            #     vector_id = f"{story_id}__full"
            #     vectors_to_upsert.append({
            #         'id': vector_id,
            #         'values': embedding,
            #         'metadata': {
            #             'story_id': story_id,
            #             'text': combined_text,
            #             'filename': story.filename
            #         }
            #     })


            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone")
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
    
    def _keyword_search(self, query: str, field_values: List[str]) -> float:
        """Perform keyword search with fuzzy matching"""
        if not field_values or not query.strip():
            return 0.0
        
        # Remove stop words and short words
        stop_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must',
            'story', 'stories', 'tale', 'book', 'novel'
        }
        
        query_words = [
            word.strip().lower() 
            for word in re.split(r'[,\s]+', query) 
            if word.strip() and len(word.strip()) >= 3 and word.strip().lower() not in stop_words
        ]
        
        if not query_words:
            return 0.0
        
        total_score = 0.0
        field_text = ' '.join(field_values).lower()
        
        for query_word in query_words:
            best_score = 0.0
            
            # Check exact match
            if query_word in field_text.split():
                best_score = 1.0
            else:
                # Fuzzy matching
                for field_value in field_values:
                    fuzzy_score = fuzz.ratio(query_word, field_value.lower()) / 100.0
                    if fuzzy_score >= 0.8:
                        best_score = max(best_score, fuzzy_score * 0.8)
            
            total_score += best_score
        
        return total_score / len(query_words)
    
    def _semantic_search_local(self, query: str, story_id: str, field: str) -> float:
        """Local semantic search when Pinecone is not available"""
        story = self.stories.get(story_id)
        if not story:
            return 0.0
        
        field_values = getattr(story, field, [])
        if not field_values:
            return 0.0
        
        try:
            # Create embeddings for comparison
            query_embedding = self.model.encode([query])
            field_text = ' '.join(field_values)
            field_embedding = self.model.encode([field_text])
            
            # Calculate cosine similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(query_embedding, field_embedding)[0][0]
            
            # Convert to Python float and ensure bounds
            similarity = float(similarity)
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Local semantic search failed for {story_id}.{field}: {e}")
            return 0.0
    
    # def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
    #     """Search stories using hybrid approach"""
    #     if not query.strip():
    #         return []
        
    #     if not self.stories:
    #         logger.warning("No stories loaded in memory")
    #         return []
        
    #     results = {}
        
    #     # 1. Semantic search using Pinecone (if available) or local embeddings
    #     if self.index:
    #         try:
    #             query_embedding = self.model.encode([query])
    #             query_embedding = query_embedding[0].tolist()

    #             # Search in Pinecone with higher top_k to get more results
    #             search_results = self.index.query(
    #                 vector=query_embedding,
    #                 top_k=min(1000, top_k * len(self.semantic_fields)),  # Get more results,
    #                 include_metadata=True
    #             )
                
    #             field_argmax = {}
    #             for match in search_results['matches']:
    #                 field = match['metadata']['field']
    #                 story_id = match['metadata']['story_id']
    #                 if story_id not in field_argmax:
    #                     field_argmax[story_id] = {}
    #                 field_argmax[story_id][field] = 0


    #             # Process semantic search results
    #             for match in search_results['matches']:
    #                 story_id = match['metadata']['story_id']
    #                 field = match['metadata']['field']
    #                 score = match['score']

    #                 # Only include if story exists in local memory
    #                 if story_id in self.stories and score >= 0.6:
    #                     if match['score'] >= 0.75:
    #                         print(match)
    #                     # print(story_id, field, score)
    #                     if story_id not in results:
    #                         results[story_id] = {
    #                             'story': self.stories[story_id],
    #                             'scores': {},
    #                             'total_score': 0.0
    #                         }
    #                     results[story_id]['scores'][field] = max(field_argmax[story_id][field],float(match['score']))
    #                     field_argmax[story_id][field] = max(field_argmax[story_id][field],float(match['score']))

    #                     results[story_id]['total_score'] = max(results[story_id]['total_score'], results[story_id]['scores'][field])

                
                        


    #         except Exception as e:
    #             logger.error(f"Pinecone search failed: {e}")
    #     else:
    #         # Fallback to local semantic search
    #         for story_id, story in self.stories.items():
    #             if story_id not in results:
    #                 results[story_id] = {
    #                     'story': story,
    #                     'scores': {},
    #                     'total_score': 0.0
    #                 }
                
    #             for field in self.semantic_fields:
    #                 score = self._semantic_search_local(query, story_id, field)
    #                 # Ensure score is a Python float
    #                 score = float(score) if score is not None else 0.0
    #                 results[story_id]['scores'][field] = score
    #                 results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
        
    #     # 2. Keyword search for character and keyword fields
    #     for story_id, story in self.stories.items():
    #         if story_id not in results:
    #             results[story_id] = {
    #                 'story': story,
    #                 'scores': {},
    #                 'total_score': 0.0
    #             }
            
    #         for field in self.keyword_fields:
    #             field_values = getattr(story, field)
    #             score = self._keyword_search(query, field_values)
                
    #             # Ensure score is a Python float
    #             score = float(score) if score is not None else 0.0
    #             results[story_id]['scores'][field] = score
    #             results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
        
    #     # Convert to SearchResult objects and sort
    #     search_results = []
    #     for story_id, result_data in results.items():
    #         if result_data['total_score'] > 0:
    #             search_results.append(SearchResult(
    #                 story=result_data['story'],
    #                 score=result_data['total_score'],
    #                 matched_fields=result_data['scores']
    #             ))
        
    #     # Sort by score
    #     search_results.sort(key=lambda x: x.score, reverse=True)
        
    #     return search_results[:top_k]

    #     # if self.index:
    #     #     try:
    #     #         # Create query embedding
    #     #         query_embedding = self.model.encode([query])[0].tolist()

    #     #         # Search Pinecone (now each story only has one vector)
    #     #         search_results = self.index.query(
    #     #             vector=query_embedding,
    #     #             top_k=top_k,
    #     #             include_metadata=True
    #     #         )
    #     #         print(search_results)
    #     #         # Process results
    #     #         for match in search_results['matches']:
    #     #             story_id = match['metadata']['story_id']
    #     #             score = match['score']

    #     #             # Only include if story exists locally and above threshold
    #     #             if story_id in self.stories and score >= 0:
    #     #                 results[story_id] = {
    #     #                     'story': self.stories[story_id],
    #     #                     'scores': {'full_story': score},
    #     #                     'total_score': score
    #     #                 }

    #     #     except Exception as e:
    #     #         logger.error(f"Pinecone search failed: {e}")

    #     # else:
    #     #     # Fallback to local semantic search
    #     #     for story_id, story in self.stories.items():
    #     #         combined_text = getattr(story, 'combined_semantic_text', None)
    #     #         if not combined_text:
    #     #             continue

    #     #         score = float(self._semantic_search_local_full(query, combined_text))
    #     #         if score >= 0.6:
    #     #             results[story_id] = {
    #     #                 'story': story,
    #     #                 'scores': {'full_story': score},
    #     #                 'total_score': score
    #     #             }

    #     # # Keyword search for character and keyword fields (optional boost)
    #     # for story_id, story in self.stories.items():
    #     #     if story_id not in results:
    #     #         results[story_id] = {
    #     #             'story': story,
    #     #             'scores': {},
    #     #             'total_score': 0.0
    #     #         }
            
    #     #     for field in self.keyword_fields:
    #     #         field_values = getattr(story, field)
    #     #         score = float(self._keyword_search(query, field_values) or 0.0)
    #     #         results[story_id]['scores'][field] = score
    #     #         results[story_id]['total_score'] += self.weights.get(field, 0.0) * score

    #     # # Convert to SearchResult objects and sort
    #     # search_results = []
    #     # for story_id, result_data in results.items():
    #     #     if result_data['total_score'] > 0:
    #     #         search_results.append(SearchResult(
    #     #             story=result_data['story'],
    #     #             score=result_data['total_score'],
    #     #             matched_fields=result_data['scores']
    #     #         ))

    #     # search_results.sort(key=lambda x: x.score, reverse=True)

    #     # return search_results[:top_k]

    # def search_single_concept(self, concept: str, top_k: int = 100) -> Dict[str, Dict]:
    #     """Search for a single concept and return results dictionary"""
    #     if not concept.strip():
    #         return {}
        
    #     results = {}
        
    #     # 1. Semantic search using Pinecone (if available) or local embeddings
    #     if self.index:
    #         try:
    #             query_embedding = self.model.encode([concept])
    #             query_embedding = query_embedding[0].tolist()

    #             # Search in Pinecone
    #             search_results = self.index.query(
    #                 vector=query_embedding,
    #                 top_k=min(1000, top_k * len(self.semantic_fields)),
    #                 include_metadata=True
    #             )
                
    #             field_argmax = {}
    #             for match in search_results['matches']:
    #                 field = match['metadata']['field']
    #                 story_id = match['metadata']['story_id']
    #                 if story_id not in field_argmax:
    #                     field_argmax[story_id] = {}
    #                 field_argmax[story_id][field] = 0

    #             # Process semantic search results
    #             for match in search_results['matches']:
    #                 story_id = match['metadata']['story_id']
    #                 field = match['metadata']['field']
    #                 score = match['score']

    #                 # Only include if story exists in local memory
    #                 if story_id in self.stories and score >= 0.6:
    #                     if story_id not in results:
    #                         results[story_id] = {
    #                             'story': self.stories[story_id],
    #                             'scores': {},
    #                             'total_score': 0.0
    #                         }
    #                     results[story_id]['scores'][field] = max(field_argmax[story_id][field], float(score))
    #                     field_argmax[story_id][field] = max(field_argmax[story_id][field], float(score))
    #                     results[story_id]['total_score'] = max(results[story_id]['total_score'], results[story_id]['scores'][field])

    #         except Exception as e:
    #             logger.error(f"Pinecone search failed for concept '{concept}': {e}")
    #     else:
    #         # Fallback to local semantic search
    #         for story_id, story in self.stories.items():
    #             if story_id not in results:
    #                 results[story_id] = {
    #                     'story': story,
    #                     'scores': {},
    #                     'total_score': 0.0
    #                 }
                
    #             for field in self.semantic_fields:
    #                 score = self._semantic_search_local(concept, story_id, field)
    #                 score = float(score) if score is not None else 0.0
    #                 results[story_id]['scores'][field] = score
    #                 results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
        
    #     # 2. Keyword search for character and keyword fields
    #     for story_id, story in self.stories.items():
    #         if story_id not in results:
    #             results[story_id] = {
    #                 'story': story,
    #                 'scores': {},
    #                 'total_score': 0.0
    #             }
            
    #         for field in self.keyword_fields:
    #             field_values = getattr(story, field)
    #             score = self._keyword_search(concept, field_values)
    #             score = float(score) if score is not None else 0.0
    #             results[story_id]['scores'][field] = score
    #             results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
    #     with open(f"temp_{concept}.json", 'w', encoding='utf-8') as f:
    #         json.dump(results, f, default=lambda o: o.__dict__, indent=2)
    #     return results


    # def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
    #     """Search stories using hybrid approach with query decomposition"""
    #     if not query.strip():
    #         return []
        
    #     if not self.stories:
    #         logger.warning("No stories loaded in memory")
    #         return []
        
    #     # Decompose query into concepts
    #     concepts = decompose_query(query)
    #     logger.info(f"Decomposed query '{query}' into concepts: {concepts}")
        
    #     # If only one concept, use original logic
    #     if len(concepts) == 1:
    #         concept_results = self.search_single_concept(concepts[0], top_k)
    #     else:
    #         # Search for each concept separately
    #         all_concept_results = {}
            
    #         for concept in concepts:
    #             concept_results_single = self.search_single_concept(concept, top_k * 3)
                
    #             for story_id, result_data in concept_results_single.items():
    #                 if result_data['total_score'] > 0:  # Only include stories with positive scores
    #                     if story_id not in all_concept_results:
    #                         all_concept_results[story_id] = {
    #                             'story': result_data['story'],
    #                             'concept_scores': {},  # Track scores per concept
    #                             'total_score': 0.0,
    #                             'concepts_matched': 0
    #                         }
                        
    #                     # Store the best score for this concept
    #                     all_concept_results[story_id]['concept_scores'][concept] = result_data['total_score']
    #                     all_concept_results[story_id]['concepts_matched'] += 1
            
    #         # Calculate final scores with coverage bonus
    #         for story_id, result_data in all_concept_results.items():
    #             concept_scores = list(result_data['concept_scores'].values())
    #             concepts_matched = result_data['concepts_matched']
                
    #             # Base score: average of concept scores
    #             base_score = sum(concept_scores) / len(concept_scores)
                
    #             # Coverage bonus: boost stories matching multiple concepts
    #             coverage_bonus = 1.0
    #             if concepts_matched > 1:
    #                 coverage_bonus = 1.0 + (concepts_matched - 1) * 0.3  # 30% bonus per additional concept
                
    #             # Final score
    #             result_data['total_score'] = base_score * coverage_bonus
                
    #             # Store matched fields info (combine from all concepts)
    #             result_data['matched_fields'] = {}
    #             for concept, score in result_data['concept_scores'].items():
    #                 result_data['matched_fields'][f"concept_{concept}"] = score
            
    #         concept_results = all_concept_results
        
    #     # Convert to SearchResult objects and sort
    #     search_results = []
    #     for story_id, result_data in concept_results.items():
    #         if result_data['total_score'] > 0:
    #             # Handle both single concept and multi-concept results
    #             matched_fields = result_data.get('matched_fields', result_data.get('scores', {}))
                
    #             search_results.append(SearchResult(
    #                 story=result_data['story'],
    #                 score=result_data['total_score'],
    #                 matched_fields=matched_fields
    #             ))
        
    #     # Sort by score
    #     search_results.sort(key=lambda x: x.score, reverse=True)
        
    #     return search_results[:top_k]
    
    def search_single_concept(self, concept: str, top_k: int = 100) -> Dict[str, Dict]:
        """Search for a single concept and return results dictionary"""
        if not concept.strip():
            return {}
        
        results = {}
        
        # 1. Semantic search using Pinecone (if available) or local embeddings
        if self.index:
            try:
                query_embedding = self.model.encode([concept])
                query_embedding = query_embedding[0].tolist()

                # Search in Pinecone
                search_results = self.index.query(
                    vector=query_embedding,
                    top_k=min(1000, top_k * len(self.semantic_fields)),
                    include_metadata=True
                )
                
                field_argmax = {}
                for match in search_results['matches']:
                    field = match['metadata']['field']
                    story_id = match['metadata']['story_id']
                    if story_id not in field_argmax:
                        field_argmax[story_id] = {}
                    field_argmax[story_id][field] = 0

                # Process semantic search results
                for match in search_results['matches']:
                    story_id = match['metadata']['story_id']
                    field = match['metadata']['field']
                    score = match['score']

                    # Only include if story exists in local memory
                    if story_id in self.stories and score >= 0.6:
                        if story_id not in results:
                            results[story_id] = {
                                'story': self.stories[story_id],
                                'scores': {},
                                'total_score': 0.0,
                                'max_score': 0.0  # Track the highest single score for this concept
                            }
                        results[story_id]['scores'][field] = max(field_argmax[story_id][field], float(score))
                        field_argmax[story_id][field] = max(field_argmax[story_id][field], float(score))
                        results[story_id]['total_score'] = max(results[story_id]['total_score'], results[story_id]['scores'][field])
                        # Track the maximum score across all fields for this story
                        results[story_id]['max_score'] = max(results[story_id]['max_score'], float(score))

            except Exception as e:
                logger.error(f"Pinecone search failed for concept '{concept}': {e}")
        else:
            # Fallback to local semantic search
            for story_id, story in self.stories.items():
                if story_id not in results:
                    results[story_id] = {
                        'story': story,
                        'scores': {},
                        'total_score': 0.0,
                        'max_score': 0.0
                    }
                
                max_field_score = 0.0
                for field in self.semantic_fields:
                    score = self._semantic_search_local(concept, story_id, field)
                    score = float(score) if score is not None else 0.0
                    results[story_id]['scores'][field] = score
                    results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
                    max_field_score = max(max_field_score, score)
                
                results[story_id]['max_score'] = max_field_score
        
        # 2. Keyword search for character and keyword fields
        for story_id, story in self.stories.items():
            if story_id not in results:
                results[story_id] = {
                    'story': story,
                    'scores': {},
                    'total_score': 0.0,
                    'max_score': 0.0
                }
            
            for field in self.keyword_fields:
                field_values = getattr(story, field)
                score = self._keyword_search(concept, field_values)
                score = float(score) if score is not None else 0.0
                results[story_id]['scores'][field] = score
                results[story_id]['total_score'] += self.weights.get(field, 0.0) * score
                # Update max_score if this keyword score is higher
                results[story_id]['max_score'] = max(results[story_id]['max_score'], score)
        os.makedirs("temp", exist_ok=True)
        with open(f"temp/{concept}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, default=lambda o: o.__dict__, indent=2)
        return results


    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """Search stories using hybrid approach with query decomposition"""
        if not query.strip():
            return []
        
        if not self.stories:
            logger.warning("No stories loaded in memory")
            return []
        
        # Decompose query into concepts
        concepts = decompose_query(query)
        logger.info(f"Decomposed query '{query}' into concepts: {concepts}")
        
        # If only one concept, use original logic
        if len(concepts) == 1:
            concept_results = self.search_single_concept(concepts[0], top_k)
        else:
            # Search for each concept separately with expanded top_k
            all_concept_results = {}
            
            for concept in concepts:
                # Search with top_k * 5 to get more candidates for intersection
                concept_results_single = self.search_single_concept(concept, top_k * 5)
                all_concept_results[concept] = concept_results_single
            
            # Find intersection of stories across all concepts
            # Start with stories from first concept
            first_concept = concepts[0]
            intersected_stories = set(all_concept_results[first_concept].keys())
            
            # Keep only stories that appear in ALL concept searches
            for concept in concepts[1:]:
                intersected_stories = intersected_stories.intersection(
                    set(all_concept_results[concept].keys())
                )
            
            logger.info(f"Found {len(intersected_stories)} stories in intersection of all {len(concepts)} concepts")
            
            # Calculate final scores by multiplying concept scores together
            final_results = {}
            for story_id in intersected_stories:
                # Get the max score for each concept for this story
                concept_max_scores = []
                story_data = None
                
                for concept in concepts:
                    concept_result = all_concept_results[concept][story_id]
                    # Use the max_score (highest relevance across all fields) for this concept
                    concept_max_scores.append(concept_result['max_score'])
                    if story_data is None:
                        story_data = concept_result['story']
                
                # Only include stories where all concepts have positive scores
                if all(score > 0 for score in concept_max_scores):
                    # Multiply the max scores together for final relevance
                    multiplied_score = 1.0
                    for score in concept_max_scores:
                        multiplied_score *= score
                    
                    final_results[story_id] = {
                        'story': story_data,
                        'total_score': multiplied_score,
                        'concept_scores': dict(zip(concepts, concept_max_scores)),
                        'matched_fields': {f"concept_{concept}": score for concept, score in zip(concepts, concept_max_scores)}
                    }
            
            concept_results = final_results
        
        # Convert to SearchResult objects and sort
        search_results = []
        for story_id, result_data in concept_results.items():
            if result_data['total_score'] > 0:
                # Handle both single concept and multi-concept results
                matched_fields = result_data.get('matched_fields', result_data.get('scores', {}))
                
                search_results.append(SearchResult(
                    story=result_data['story'],
                    score=result_data['total_score'],
                    matched_fields=matched_fields
                ))
        
        # Sort by score (highest multiplied scores first)
        search_results.sort(key=lambda x: x.score, reverse=True)
        
        return search_results[:top_k]
    
    def get_stats(self):
        """Get current statistics"""
        return {
            'total_stories': len(self.stories),
            'pinecone_connected': self.index is not None,
            'embedding_provider': self.model.provider,
            'embedding_dimension': self.model.dimension
        }

class FeedbackDB:
    def __init__(self, supabase_url: str = None, supabase_key: str = None):
        """
        Initialize Supabase client
        Args:
            supabase_url: Supabase project URL (or set SUPABASE_URL env var)
            supabase_key: Supabase anon key (or set SUPABASE_ANON_KEY env var)
        """
        self.supabase_url = supabase_url or os.getenv('SUPABASE_URL')
        self.supabase_key = supabase_key or os.getenv('SUPABASE_ANON_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key must be provided either as parameters or environment variables")
        
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.init_db()
    
    def init_db(self):
        """
        Initialize the feedback table in Supabase
        Note: You should run this SQL in Supabase SQL Editor instead:
        
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            query TEXT NOT NULL,
            story_id TEXT NOT NULL,
            feedback_text TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            user_ip TEXT
        );
        """
        # Check if table exists by trying to select from it
        try:
            result = self.supabase.table('feedback').select('id').limit(1).execute()
            print("Feedback table exists")
        except Exception as e:
            print(f"Warning: Could not verify feedback table exists. Error: {e}")
            print("Please create the table manually in Supabase SQL Editor using the SQL in the docstring above.")
    
    def save_feedback(self, query: str, story_id: str, feedback_text: str, user_ip: Optional[str] = None) -> bool:
        """
        Save feedback to Supabase
        Returns True if successful, False otherwise
        """
        try:
            data = {
                'query': query,
                'story_id': story_id,
                'feedback_text': feedback_text,
                'user_ip': user_ip,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            result = self.supabase.table('feedback').insert(data).execute()
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error saving feedback: {e}")
            return False
    
    def get_all_feedback(self) -> List[Dict]:
        """
        Get all feedback records ordered by timestamp (newest first)
        Returns list of dictionaries
        """
        try:
            result = self.supabase.table('feedback').select('*').order('timestamp', desc=True).execute()
            return result.data
            
        except Exception as e:
            print(f"Error getting feedback: {e}")
            return []
    
    def get_feedback_by_story_id(self, story_id: str) -> List[Dict]:
        """Get feedback for a specific story"""
        try:
            result = self.supabase.table('feedback').select('*').eq('story_id', story_id).order('timestamp', desc=True).execute()
            return result.data
            
        except Exception as e:
            print(f"Error getting feedback by story_id: {e}")
            return []
    
    def get_recent_feedback(self, limit: int = 50) -> List[Dict]:
        """Get recent feedback with limit"""
        try:
            result = self.supabase.table('feedback').select('*').order('timestamp', desc=True).limit(limit).execute()
            return result.data
            
        except Exception as e:
            print(f"Error getting recent feedback: {e}")
            return []
    
    def delete_feedback(self, feedback_id: int) -> bool:
        """Delete feedback by ID"""
        try:
            result = self.supabase.table('feedback').delete().eq('id', feedback_id).execute()
            return len(result.data) > 0
            
        except Exception as e:
            print(f"Error deleting feedback: {e}")
            return False
        
# Initialize feedback DB after search_engine initialization
feedback_db = FeedbackDB()

# Initialize search engine
search_engine = StorySearchEngine()

# Initialize voice handler
voice_handler = VoiceHandler()

# HTML Template
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Story Search Engine</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .loading { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .fade-in { animation: fadeIn 0.5s ease-in; }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body class="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
    <div class="max-w-6xl mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <div class="flex items-center justify-center mb-4">
                <svg class="w-12 h-12 text-blue-600 mr-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
                </svg>
                <h1 class="text-4xl font-bold text-gray-900">Story Search Engine</h1>
            </div>
            <p class="text-xl text-gray-600">Find the perfect story using AI-powered semantic search</p>
            
            <!-- API Status -->
            <div id="api-status" class="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium mt-4">
                <div id="status-indicator" class="w-2 h-2 rounded-full mr-2"></div>
                <span id="status-text">Checking API...</span>
            </div>
        </div>

        <!-- Upload Section -->
        <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
            <h2 class="text-2xl font-semibold text-gray-900 mb-4 flex items-center">
                <svg class="w-6 h-6 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                </svg>
                Upload Story Data
            </h2>
            <form id="upload-form" class="flex items-center space-x-4">
                <div class="flex-1">
                    <input type="file" id="csv-file" accept=".csv" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100">
                </div>
                <button type="submit" id="upload-btn" disabled class="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center">
                    <svg id="upload-icon" class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                    <span id="upload-text">Upload CSV</span>
                </button>
            </form>

            <!-- Sync from Pinecone Button -->
            <div class="mt-4 pt-4 border-t border-gray-200">
                <button id="sync-btn" class="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center text-sm">
                    <svg id="sync-icon" class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                    </svg>
                    <span id="sync-text">Sync from Pinecone</span>
                </button>
                <p class="text-xs text-gray-500 mt-1">Load stories from existing Pinecone vectors if available</p>
            </div>
        </div>

        <!-- CSV Schema Documentation -->
        <div class="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h3 class="text-lg font-semibold text-blue-900 mb-2">CSV Schema Required</h3>
            <p class="text-sm text-blue-800 mb-2">Your CSV file must contain these exact columns:</p>
            <div class="grid grid-cols-2 gap-4 text-xs">
                <div>
                    <strong>Required Columns:</strong>
                    <ul class="list-disc list-inside text-blue-700 mt-1">
                        <li>filename</li>
                        <li>character_primary</li>
                        <li>character_secondary</li>
                        <li>setting_primary</li>
                        <li>setting_secondary</li>
                        <li>theme_primary</li>
                        <li>theme_secondary</li>
                    </ul>
                </div>
                <div>
                    <strong>Additional Columns:</strong>
                    <ul class="list-disc list-inside text-blue-700 mt-1">
                        <li>events_primary</li>
                        <li>events_secondary</li>
                        <li>emotions_primary</li>
                        <li>emotions_secondary</li>
                        <li>keywords</li>
                    </ul>
                </div>
            </div>
            <p class="text-xs text-blue-600 mt-2">
                <strong>Format:</strong> List fields should be formatted as Python lists: <code>["item1", "item2", "item3"]</code>
            </p>
        </div>

        <!-- Search Section -->
        <div class="bg-white rounded-xl shadow-lg p-6 mb-8">
            <form id="search-form" class="space-y-4">
                <div class="flex items-center space-x-4">
                    <div class="flex-1 relative">
                        <svg class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                        <input type="text" id="search-input" placeholder="Search for stories... (e.g., 'friendship underwater animals', 'robot school adventure')" class="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg">
                    </div>
                    <button type="submit" id="search-btn" class="px-8 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center text-lg font-medium">
                        <svg id="search-icon" class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
                        </svg>
                        <span id="search-text">Search</span>
                    </button>
                </div>
            </form>

            <!-- Voice Search Button -->
            <div class="mt-4 flex items-center justify-center space-x-4">
                <button id="voice-search-btn" class="flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed">
                    <svg id="voice-icon" class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path>
                    </svg>
                    <span id="voice-text">Voice Search</span>
                </button>
                
                <button id="view-feedback-btn" class="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-1.586l-4.707 4.707z"></path>
                    </svg>
                    View All Feedback
                </button>
                <button id="delete-stories-btn" class="flex items-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700">
                    <svg class="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                    </svg>
                    Delete Stories
                </button>
            </div>

            <!-- Audio input (hidden) -->
            <input type="file" id="audio-input" accept="audio/*" style="display: none;">

            <!-- Example Queries -->
            <div class="mt-4">
                <p class="text-sm text-gray-600 mb-2">Try these example searches:</p>
                <div class="flex flex-wrap gap-2">
                    <button class="example-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors" data-query="friendship underwater animals">friendship underwater animals</button>
                    <button class="example-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors" data-query="robot artificial intelligence school">robot artificial intelligence school</button>
                    <button class="example-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors" data-query="adventure jungle betrayal">adventure jungle betrayal</button>
                    <button class="example-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors" data-query="fear anxiety emotions">fear anxiety emotions</button>
                    <button class="example-btn px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors" data-query="children magic fairy tale">children magic fairy tale</button>
                </div>
            </div>
        </div>

        <!-- Error Display -->
        <div id="error-display" class="hidden bg-red-50 border border-red-200 rounded-lg p-4 mb-6 flex items-center">
            <svg class="w-5 h-5 text-red-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span id="error-text" class="text-red-700"></span>
        </div>

        <!-- Results Section -->
        <div id="results-section" class="hidden space-y-6">
            <h2 class="text-2xl font-semibold text-gray-900 flex items-center">
                <svg class="w-6 h-6 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
                </svg>
                Search Results (<span id="results-count">0</span>)
            </h2>
            <div id="results-container"></div>
        </div>

        <!-- No Results -->
        <div id="no-results" class="hidden text-center py-12">
            <svg class="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"></path>
            </svg>
            <h3 class="text-xl font-medium text-gray-900 mb-2">No stories found</h3>
            <p class="text-gray-600">Try different keywords or upload more story data</p>
        </div>

        <!-- Footer -->
        <footer class="mt-16 text-center text-gray-500 text-sm">
            <p>Story Search Engine powered by Sentence Transformers and Pinecone</p>
        </footer>
    </div>

    <script>
        // Global variables
        let isSearching = false;
        let isUploading = false;
        let isSyncing = false;
        // DOM elements
        const searchForm = document.getElementById('search-form');
        const searchInput = document.getElementById('search-input');
        const searchBtn = document.getElementById('search-btn');
        const searchIcon = document.getElementById('search-icon');
        const searchText = document.getElementById('search-text');
        const uploadForm = document.getElementById('upload-form');
        const csvFileInput = document.getElementById('csv-file');
        const uploadBtn = document.getElementById('upload-btn');
        const uploadIcon = document.getElementById('upload-icon');
        const uploadText = document.getElementById('upload-text');
        const syncBtn = document.getElementById('sync-btn');
        const syncIcon = document.getElementById('sync-icon');
        const syncText = document.getElementById('sync-text');
        const errorDisplay = document.getElementById('error-display');
        const errorText = document.getElementById('error-text');
        const resultsSection = document.getElementById('results-section');
        const resultsContainer = document.getElementById('results-container');
        const resultsCount = document.getElementById('results-count');
        const noResults = document.getElementById('no-results');
        const apiStatus = document.getElementById('api-status');
        const statusIndicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('status-text');

        // Helper functions
        function showError(message) {
            errorText.textContent = message;
            errorDisplay.classList.remove('hidden');
            setTimeout(() => {
                errorDisplay.classList.add('hidden');
            }, 5000);
        }

        function hideError() {
            errorDisplay.classList.add('hidden');
        }

        function updateApiStatus(status) {
            if (status.status === 'healthy') {
                apiStatus.className = 'inline-flex items-center px-4 py-2 rounded-full text-sm font-medium mt-4 bg-green-100 text-green-800';
                statusIndicator.className = 'w-2 h-2 rounded-full mr-2 bg-green-400';
                statusText.textContent = `API: ${status.status} | Stories: ${status.stories_loaded || 0} | Pinecone: ${status.pinecone_connected ? 'Connected' : 'Disconnected'}`;
            } else {
                apiStatus.className = 'inline-flex items-center px-4 py-2 rounded-full text-sm font-medium mt-4 bg-red-100 text-red-800';
                statusIndicator.className = 'w-2 h-2 rounded-full mr-2 bg-red-400';
                statusText.textContent = `API: ${status.status || 'unhealthy'}`;
            }
        }

        function getScoreColor(score) {
            if (score >= 0.8) return 'text-green-600 bg-green-50';
            if (score >= 0.6) return 'text-blue-600 bg-blue-50';
            if (score >= 0.4) return 'text-yellow-600 bg-yellow-50';
            return 'text-gray-600 bg-gray-50';
        }

        function formatFieldName(fieldName) {
            return fieldName.replace('_', ' ').replace(/\b\w/g, function(l) { return l.toUpperCase(); });
        }

        function getFieldIcon(fieldName) {
            const icons = {
                character_primary: '👥',
                character_secondary: '👤',
                setting_primary: '📍',
                setting_secondary: '🗺️',
                theme_primary: '⭐',
                theme_secondary: '✨',
                emotions_primary: '❤️',
                emotions_secondary: '💭',
                events_primary: '⚡',
                events_secondary: '🔸',
                keywords: '📚'
            };
            return icons[fieldName] || '📖';
        }

        function extractStoryId(storyId) {
            if (storyId.startsWith('data-')) {
                return storyId.substring(5);  // Removes 'data-'
            } else if (storyId.startsWith('Joyful-')) {
                return storyId.substring(7);
            } else {
                return storyId;  // Return as is if no match
            }
        }




        function renderResults(results, currentQuery = '') {
            if (!results || results.length === 0) {
                resultsSection.classList.add('hidden');
                noResults.classList.remove('hidden');
                return;
            }

            noResults.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            resultsCount.textContent = results.length;

            resultsContainer.innerHTML = results.map((result, index) => {
                const story = result.story;
                const filename = story.filename || 'Unknown Story';
                const title = filename.replace('.txt', '').replace(/-/g, ' ').replace(/\b\w/g, function(l) { return l.toUpperCase(); });
                
                // Create story details
                const storyFields = Object.entries(story).filter(([key, value]) => 
                    key !== 'id' && key !== 'filename' && Array.isArray(value) && value.length > 0
                );

                const fieldHtml = storyFields.map(([fieldName, fieldValues]) => {
                    const icon = getFieldIcon(fieldName);
                    const displayValues = fieldValues;
                    const remainingCount = fieldValues;
                    
                    return `
                        <div class="bg-gray-50 rounded-lg p-3">
                            <div class="flex items-center mb-2">
                                <span class="mr-2">${icon}</span>
                                <span class="text-sm font-medium text-gray-700">${formatFieldName(fieldName)}</span>
                            </div>
                            <div class="space-y-1">
                                ${displayValues.map(value => 
                                    `<span class="inline-block bg-white px-2 py-1 text-xs text-gray-600 rounded mr-1 mb-1">${value}</span>`
                                ).join('')}
                                ${remainingCount > 0 ? 
                                    `<span class="inline-block bg-gray-200 px-2 py-1 text-xs text-gray-500 rounded">+${remainingCount} more</span>` : ''
                                }
                            </div>
                        </div>
                    `;
                }).join('');

                // Create field scores
                const scoreHtml = result.matched_fields ? 
                    Object.entries(result.matched_fields)
                        .filter(([, score]) => score > 0)
                        .sort(([, a], [, b]) => b - a)
                        .map(([field, score]) => `
                            <div class="px-2 py-1 rounded text-xs text-center ${getScoreColor(score)}">
                                <div class="font-medium">${formatFieldName(field)}</div>
                                <div>${(score * 100).toFixed(0)}%</div>
                            </div>
                        `).join('') : '';

                return `
                    <div class="bg-white rounded-xl shadow-lg overflow-hidden fade-in">
                        <div class="p-6">
                            <div class="flex items-start justify-between mb-4">
                                <div class="flex-1">
                                    <h3 class="text-xl font-bold text-gray-900 mb-2">${title}</h3>
                                    <div class="flex items-center space-x-4 text-sm text-gray-500">
                                    <a href="https://storyweaver.org.in/en/stories/${extractStoryId(story.id)}?mode=read" target="_blank" class="hover:underline text-blue-600">
                                            Link to StoryID: ${extractStoryId(story.id) || 'N/A'}
                                        </a>

                                        <span class="px-2 py-1 rounded-full text-xs font-medium ${getScoreColor(result.score || 0)}">
                                            Match Score: ${((result.score || 0) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                                ${fieldHtml}
                            </div>
                            
                            ${scoreHtml ? `
                                <div class="border-t pt-4 mb-4">
                                    <h4 class="text-sm font-medium text-gray-700 mb-2">Field Match Scores</h4>
                                    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                                        ${scoreHtml}
                                    </div>
                                </div>
                            ` : ''}
                            
                            <!-- Feedback Section -->
                            <div class="border-t pt-4">
                                <h4 class="text-sm font-medium text-gray-700 mb-3 flex items-center">
                                    <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-1.586l-4.707 4.707z"></path>
                                    </svg>
                                    Share your feedback
                                </h4>
                                <div class="space-y-2">
                                    <textarea 
                                        id="feedback-${story.id}" 
                                        placeholder="How well does this story match your search? Any suggestions for improvement?"
                                        class="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm resize-none"
                                        rows="2"
                                    ></textarea>
                                    <div class="flex items-center justify-between">
                                        <span class="text-xs text-gray-500">Help us improve our recommendations</span>
                                        <button 
                                            onclick="submitFeedback('${story.id}', '${currentQuery}', ${index}, event)"
                                            class="px-4 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700 transition-colors"
                                        >
                                            Submit Feedback
                                        </button>
                                    </div>
                                </div>
                                <div id="feedback-status-${story.id}" class="mt-2 hidden">
                                    <div class="flex items-center text-xs text-green-600">
                                        <svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                            <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                        </svg>
                                        Thank you for your feedback!
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }

        // API functions
        async function checkHealth() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                updateApiStatus(data);
            } catch (error) {
                updateApiStatus({ status: 'unhealthy', error: error.message });
            }
        }

        async function performSearch(query) {
            if (isSearching) return;
            
            isSearching = true;
            searchBtn.disabled = true;
            searchIcon.classList.add('loading');
            searchText.textContent = 'Searching...';
            hideError();

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query.trim(),
                        top_k: 20
                    }),
                });

                if (!response.ok) {
                    throw new Error(`Search failed: ${response.statusText}`);
                }

                const data = await response.json();
                renderResults(data.results || [], query);
                
            } catch (error) {
                showError(error.message);
                renderResults([]);
            } finally {
                isSearching = false;
                searchBtn.disabled = false;
                searchIcon.classList.remove('loading');
                searchText.textContent = 'Search';
            }
        }

        async function uploadFile(file) {
            if (isUploading) return;
            
            isUploading = true;
            uploadBtn.disabled = true;
            uploadIcon.classList.add('loading');
            uploadText.textContent = 'Uploading...';
            hideError();

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) {
                    throw new Error(`Upload failed: ${response.statusText}`);
                }

                const data = await response.json();
                alert(`Success! ${data.stories_loaded} stories loaded.`);
                csvFileInput.value = '';
                checkHealth(); // Refresh health status
                
            } catch (error) {
                showError(error.message);
            } finally {
                isUploading = false;
                uploadBtn.disabled = !csvFileInput.files[0];
                uploadIcon.classList.remove('loading');
                uploadText.textContent = 'Upload CSV';
            }
        }

        async function submitFeedback(storyId, query, resultIndex, event) {
            const button = event.target;
            const originalText = button.textContent;
            try {
                const textareaId = `feedback-${storyId}`;
                const statusId = `feedback-status-${storyId}`;
                
                const textarea = document.getElementById(textareaId);
                const statusDiv = document.getElementById(statusId);
                const feedback = textarea.value.trim();
                
                if (!feedback) {
                    showError('Please enter your feedback before submitting');
                    return;
                }
                
                // Disable the button temporarily
                button.disabled = true;
                button.textContent = 'Submitting...';
                
                const response = await fetch('/submit-feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        story_id: storyId,
                        feedback: feedback
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to submit feedback');
                }
                
                // Show success message
                textarea.value = '';
                textarea.disabled = true;
                button.style.display = 'none';
                statusDiv.classList.remove('hidden');
                
                // Optional: Show a brief success message
                showSuccess('Feedback submitted successfully!');
                
            } catch (error) {
                showError('Failed to submit feedback: ' + error.message);
                // Re-enable button on error
                button.disabled = false;
                button.textContent = originalText;
            }
        }
        
        function showSuccess(message) {
            // Create or update success display
            let successDisplay = document.getElementById('success-display');
            if (!successDisplay) {
                successDisplay = document.createElement('div');
                successDisplay.id = 'success-display';
                successDisplay.className = 'fixed top-4 right-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-center z-50 shadow-lg';
                successDisplay.innerHTML = `
                    <svg class="w-5 h-5 text-green-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                    </svg>
                    <span id="success-text" class="text-green-700"></span>
                `;
                document.body.appendChild(successDisplay);
            }
            
            document.getElementById('success-text').textContent = message;
            successDisplay.classList.remove('hidden');
            
            setTimeout(() => {
                successDisplay.classList.add('hidden');
            }, 3000);
        }


        async function syncFromPinecone() {
            if (isSyncing) return;
            
            isSyncing = true;
            syncBtn.disabled = true;
            syncIcon.classList.add('loading');
            syncText.textContent = 'Syncing...';
            hideError();

            try {
                const response = await fetch('/sync', {
                    method: 'POST',
                });

                if (!response.ok) {
                    throw new Error(`Sync failed: ${response.statusText}`);
                }

                const data = await response.json();
                alert(`Success! ${data.stories_synced} stories synced from Pinecone.`);
                checkHealth(); // Refresh health status
                
            } catch (error) {
                showError(error.message);
            } finally {
                isSyncing = false;
                syncBtn.disabled = false;
                syncIcon.classList.remove('loading');
                syncText.textContent = 'Sync from Pinecone';
            }
        }

        // Event listeners
        searchForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const query = searchInput.value.trim();
            if (query) {
                performSearch(query);
            }
        });

        uploadForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const file = csvFileInput.files[0];
            if (file) {
                uploadFile(file);
            }
        });

        syncBtn.addEventListener('click', () => {
            syncFromPinecone();
        });

        csvFileInput.addEventListener('change', (e) => {
            uploadBtn.disabled = !e.target.files[0];
        });

        // Example query buttons
        document.querySelectorAll('.example-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.getAttribute('data-query');
                searchInput.value = query;
                performSearch(query);
            });
        });

        // Voice search functionality
        const voiceSearchBtn = document.getElementById('voice-search-btn');
        const voiceIcon = document.getElementById('voice-icon');
        const voiceText = document.getElementById('voice-text');
        const audioInput = document.getElementById('audio-input');
        const deleteStoriesBtn = document.getElementById('delete-stories-btn');

        let isRecording = false;
        let mediaRecorder;
        let audioChunks = [];

        voiceSearchBtn.addEventListener('click', async () => {
            if (!isRecording) {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = async () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        await sendVoiceSearch(audioBlob);
                        audioChunks = [];
                    };
                    
                    mediaRecorder.start();
                    isRecording = true;
                    voiceIcon.classList.add('animate-pulse');
                    voiceText.textContent = 'Recording... Click to stop';
                    voiceSearchBtn.classList.add('bg-red-600');
                    voiceSearchBtn.classList.remove('bg-green-600');
                    
                } catch (error) {
                    showError('Microphone access denied');
                }
            } else {
                mediaRecorder.stop();
                isRecording = false;
                voiceIcon.classList.remove('animate-pulse');
                voiceText.textContent = 'Processing...';
                voiceSearchBtn.disabled = true;
            }
        });

        async function sendVoiceSearch(audioBlob) {
            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'voice_search.wav');
                
                const response = await fetch('/voice-search', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Voice search failed');
                }
                
                const data = await response.json();
                
                // Update search input with enhanced query
                searchInput.value = data.enhanced_query;
                
                // Show results
                renderResults(data.results || [], data.enhanced_query);
                
                // Show query transformation
                if (data.original_query !== data.enhanced_query) {
                    alert(`Voice: "${data.original_query}"\nEnhanced: "${data.enhanced_query}"`);
                }
                
            } catch (error) {
                showError('Voice search failed: ' + error.message);
            } finally {
                // Reset button
                voiceText.textContent = 'Voice Search';
                voiceSearchBtn.disabled = false;
                voiceSearchBtn.classList.remove('bg-red-600');
                voiceSearchBtn.classList.add('bg-green-600');
            }
        }

        // View feedback functionality
        const viewFeedbackBtn = document.getElementById('view-feedback-btn');
        
        async function showFeedbackModal(feedbackData) {
            // Create modal
            const modal = document.createElement('div');
            modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50';
            
            const content = document.createElement('div');
            content.className = 'bg-white rounded-xl shadow-xl max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col';
            
            content.innerHTML = `
                <div class="p-6 border-b">
                    <div class="flex items-center justify-between">
                        <h2 class="text-2xl font-bold text-gray-900">All Feedback</h2>
                        <button class="text-gray-500 hover:text-gray-700" onclick="this.closest('.fixed').remove()">
                            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="p-6 overflow-auto flex-1">
                    <div class="space-y-4">
                        ${feedbackData.map(feedback => `
                            <div class="bg-gray-50 rounded-lg p-4">
                                <div class="flex items-start justify-between">
                                    <div class="space-y-1">
                                        <p class="text-sm font-medium text-gray-900">Query: "${feedback.query}"</p>
                                        <p class="text-sm text-gray-600">Story ID: ${feedback.story_id}</p>
                                        <p class="text-sm text-gray-500">${new Date(feedback.timestamp).toLocaleString()}</p>
                                    </div>
                                </div>
                                <div class="mt-2">
                                    <p class="text-gray-700">${feedback.feedback_text}</p>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
            
            modal.appendChild(content);
            document.body.appendChild(modal);
        }
        
        viewFeedbackBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/get-feedback');
                if (!response.ok) {
                    throw new Error('Failed to fetch feedback');
                }
                
                const data = await response.json();
                showFeedbackModal(data.feedback);
                
            } catch (error) {
                showError('Failed to load feedback: ' + error.message);
            }
        });

        // Delete stories functionality
        deleteStoriesBtn.addEventListener('click', async () => {
            try {
                // First, get list of stories
                const response = await fetch('/list-stories');
                const data = await response.json();
                
                if (data.stories.length === 0) {
                    alert('No stories to delete');
                    return;
                }
                
                // Show confirmation dialog with story list
                const storyList = data.stories.map(story => `${story.id}: ${story.title}`).join('\\n');
                const confirmDelete = confirm(`Delete all ${data.stories.length} stories?\n\n${storyList.substring(0, 500)}${storyList.length > 500 ? '...' : ''}`);
                
                if (confirmDelete) {
                    const storyIds = data.stories.map(story => story.id);
                    
                    const deleteResponse = await fetch('/delete-stories', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            story_ids: storyIds
                        })
                    });
                    
                    if (!deleteResponse.ok) {
                        throw new Error('Delete failed');
                    }
                    
                    const deleteData = await deleteResponse.json();
                    alert(deleteData.message);
                    
                    // Refresh page
                    location.reload();
                }
                
            } catch (error) {
                showError('Delete failed: ' + error.message);
            }
        });

        // Initialize
        checkHealth();
    </script>
</body>
</html>
"""

# Flask routes
@app.route('/')
def home():
    """Serve the main HTML page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    stats = search_engine.get_stats()
    return jsonify({
        'status': 'healthy',
        'pinecone_connected': stats['pinecone_connected'],
        'stories_loaded': stats['total_stories'],
        'embedding_provider': stats['embedding_provider'],
        'embedding_dimension': stats['embedding_dimension']
    })

def translate_to_english(text):
    """Translate text to English using Gemini"""
    try:
        # Create translation prompt
        prompt = f"""
        Translate the following text to English if from a different language, or return the original text if already in english. Only return the translation, no additional text:
        
        Text to translate: {text}
        """
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        
        return translated_text
        
    except Exception as e:
        logging.error(f"Translation error: {e}")
        # Return original text if translation fails
        return text, 'unknown'
    
import os
import logging
import google.generativeai as generative_ai

def translate_and_refine(text):
    """
    Use Gemini to:
    1. Translate input to English if necessary.
    2. Strip any story-related phrases like 'story of', 'book about', etc.
    3. Return only the refined, meaningful core query.

    Returns:
        str: Cleaned and translated query suitable for search.
    """
    try:
        prompt = f"""
You are a query cleaning assistant for a story search engine.

Instructions:
- Translate the input to English if it is in a different language.
- Then, remove common filler or generic phrases related to stories or requests. This includes:
  "story of", "stories about", "a tale on", "book of", "novel about",
  as well as conversational prefixes like "tell me", "give me", "show me", "I want a story about", etc.
- Only return the essential topic or subject of the story in a short phrase.
- Do not repeat or rephrase the original structure.
- Do not include extra words like "a", "the", "story", "book", etc.
- DO NOT explain or describe anything — just return the final cleaned phrase.

Examples:
Input: "Tell me a story of teenagers"  
Output: teenagers

Input: "Muéstrame una historia sobre la valentía"  
Output: bravery

Input: "Give me a novel about freedom fighters"  
Output: freedom fighters

Input: "A tale on friendship and loss"  
Output: friendship and loss

Now process this:
{text}
"""

        # Gemini setup
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return text.strip()

def get_font(text):
    """Translate text to English using Gemini"""
    try:
        # Create translation prompt
        prompt = f"""
        Write the text in the script it's spoken, like text in devnagri script for Hindi, text in Arabic script for Arabic, etc. If the text is already in English, return 'as it is. Only return the scripted text, no additional text:        
        Text to translate: {text}
        """
        gemini_api_key = os.getenv('GOOGLE_API_KEY')
        model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
        generative_ai.configure(api_key=gemini_api_key)
        model = generative_ai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        translated_text = response.text.strip()
        
        return translated_text
        
    except Exception as e:
        logging.error(f"Translation error: {e}")
        # Return original text if translation fails
        return text, 'unknown'
    
@app.route('/search', methods=['POST'])
def search_stories():
    """Search stories endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Query is required'}), 400
        
        query = data['query'].strip()
        top_k = data.get('top_k', 20)
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        
        # Perform search
        # query = translate_to_english(query)
        # query = refine_semantic_query(query)
        query = translate_and_refine(query)
        print(f"Search query: {query}")
        results = search_engine.search(query, top_k)
        
        # Convert results to JSON-serializable format
        response_data = []
        for result in results:
            # Convert numpy types to Python native types
            story_dict = asdict(result.story)
            score = float(result.score) if result.score is not None else 0.0
            matched_fields = {
                field: float(score) if score is not None else 0.0 
                for field, score in result.matched_fields.items()
            }
            
            response_data.append({
                'story': story_dict,
                'score': score,
                'matched_fields': matched_fields
            })
            # print(response_data)
        return jsonify({
            'query': query,
            'results': response_data,
            'total_found': len(response_data)
        })
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload', methods=['POST'])
def upload_data():
    """Upload new CSV data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'error': 'File must be a CSV'}), 400
        
        # Read file content
        csv_content = file.read().decode('utf-8')
        
        # Load new data
        search_engine.load_data(csv_data=csv_content)
        
        return jsonify({
            'message': 'Data uploaded successfully',
            'stories_loaded': len(search_engine.stories)
        })
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': 'Failed to upload data'}), 500

@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():
    """Submit user feedback for search results"""
    try:
        data = request.get_json()
        print(data)
        if not data or 'query' not in data or 'story_id' not in data or 'feedback' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Save feedback to database
        feedback_db.save_feedback(
            query=data['query'],
            story_id=data['story_id'],
            feedback_text=data['feedback'],
            user_ip=request.remote_addr
        )
        
        return jsonify({'message': 'Feedback submitted successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/voice-search', methods=['POST'])
def voice_search():
    """Handle voice search"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400
        
        # Save temporary uploaded file
        filename = secure_filename(audio_file.filename)
        raw_path = f"temp_raw_{filename}"
        audio_file.save(raw_path)

        # Convert to 16-bit mono PCM WAV using pydub
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sound = AudioSegment.from_file(raw_path)
            sound = sound.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            sound.export(tmp_wav.name, format="wav")
            converted_path = tmp_wav.name

        try:
            # Convert speech to text
            voice_query = voice_handler.speech_to_text(converted_path)

            # Enhance query with Gemini
            # enhanced_query = voice_handler.enhance_query_with_gemini(voice_query)
            enhanced_query = voice_query
            query = enhanced_query
            f_query = get_font(query)
            print(f"Voice query: (translated: {f_query})")
            query = translate_to_english(query)
            print(f"Search query: {query}")
            # Perform search
            results = search_engine.search(query, 20)

            # Prepare response
            response_data = []
            for result in results:
                story_dict = asdict(result.story)
                score = float(result.score) if result.score is not None else 0.0
                matched_fields = {
                    field: float(score) if score is not None else 0.0
                    for field, score in result.matched_fields.items()
                }
                response_data.append({
                    'story': story_dict,
                    'score': score,
                    'matched_fields': matched_fields
                })

            return jsonify({
                'original_query': voice_query,
                'enhanced_query': enhanced_query,
                'results': response_data,
                'total_found': len(response_data)
            })

        finally:
            # Cleanup
            if os.path.exists(raw_path):
                os.remove(raw_path)
            if os.path.exists(converted_path):
                os.remove(converted_path)

    except Exception as e:
        logger.error(f"Voice search error: {e}")
        return jsonify({'error': 'Voice search failed'}), 500

@app.route('/sync', methods=['POST'])
def sync_from_pinecone():
    """Sync stories from Pinecone"""
    try:
        if not search_engine.index:
            return jsonify({'error': 'Pinecone not connected'}), 400
        
        stories_before = len(search_engine.stories)
        search_engine._sync_from_pinecone()
        stories_after = len(search_engine.stories)
        
        return jsonify({
            'message': 'Sync completed successfully',
            'stories_synced': stories_after - stories_before,
            'total_stories': stories_after
        })
        
    except Exception as e:
        logger.error(f"Sync error: {e}")
        return jsonify({'error': 'Failed to sync from Pinecone'}), 500

@app.route('/delete-stories', methods=['POST'])
def delete_stories():
    """Delete stories from Pinecone"""
    try:
        if not search_engine.index:
            return jsonify({'error': 'Pinecone not connected'}), 400
        
        data = request.get_json()
        story_ids = data.get('story_ids', [])
        
        if not story_ids:
            return jsonify({'error': 'No story IDs provided'}), 400
        
        # Delete from Pinecone
        vectors_to_delete = []
        for story_id in story_ids:
            for field in search_engine.semantic_fields:
                vectors_to_delete.append(f"{story_id}_{field}")
        
        if vectors_to_delete:
            search_engine.index.delete(ids=vectors_to_delete)
        
        # Delete from local storage
        deleted_count = 0
        for story_id in story_ids:
            if story_id in search_engine.stories:
                del search_engine.stories[story_id]
                deleted_count += 1
        
        return jsonify({
            'message': f'Successfully deleted {deleted_count} stories',
            'deleted_count': deleted_count,
            'remaining_stories': len(search_engine.stories)
        })
        
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return jsonify({'error': 'Failed to delete stories'}), 500

@app.route('/list-stories', methods=['GET'])
def list_stories():
    """List all available stories"""
    try:
        stories_list = []
        for story_id, story in search_engine.stories.items():
            stories_list.append({
                'id': story_id,
                'filename': story.filename,
                'title': story.filename.replace('.txt', '').replace('-', ' ').title()
            })
        
        return jsonify({
            'stories': stories_list,
            'total_count': len(stories_list)
        })
        
    except Exception as e:
        logger.error(f"List stories error: {e}")
        return jsonify({'error': 'Failed to list stories'}), 500
    
@app.route('/get-feedback', methods=['GET'])
def get_feedback():
    """Get all feedback records"""
    try:
        # Get feedback from Supabase using the feedback_db instance
        feedback_records = feedback_db.get_all_feedback()
        
        # Transform the data to include all necessary fields
        formatted_feedback = []
        for record in feedback_records:
            formatted_feedback.append({
                'id': record.get('id'),
                'query': record.get('query', ''),
                'story_id': record.get('story_id', ''),
                'feedback_text': record.get('feedback_text', ''),
                'timestamp': record.get('timestamp', ''),
                'user_ip': record.get('user_ip', '')
            })
        
        return jsonify({'feedback': formatted_feedback})
    except Exception as e:
        print(f"Error fetching feedback: {str(e)}")  # For debugging
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)