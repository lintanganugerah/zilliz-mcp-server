"""Embedding client with auto-fallback support (OpenRouter → FastEmbed)."""

import logging
import os
import requests
from typing import List, Optional
from zilliz_mcp_server.settings import config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings with auto-fallback support."""
    
    def __init__(self):
        """Initialize embedding client."""
        self.current_method = config.embedding_method
        self.auto_fallback = config.auto_fallback_to_local
        
        # Remote (OpenRouter) config
        self.openrouter_api_key = config.openrouter_api_key
        self.openrouter_model = config.openrouter_embedding_model
        self.remote_dimension = config.remote_embedding_dimension
        self.openrouter_base_url = "https://openrouter.ai/api/v1"
        
        # Local (FastEmbed) config
        self.local_model = config.local_embedding_model
        self.local_dimension = config.local_embedding_dimension
        self.fastembed_cache_dir = config.fastembed_cache_dir
        self._local_model_instance = None
        
        logger.info(f"Embedding client initialized: method={self.current_method}, auto_fallback={self.auto_fallback}")
    
    def _get_local_model(self):
        """Lazy load local FastEmbed model."""
        if self._local_model_instance is None:
            try:
                from fastembed import TextEmbedding
                
                # Set cache directory
                os.environ["FASTEMBED_CACHE_PATH"] = self.fastembed_cache_dir
                
                logger.info(f"Loading local embedding model: {self.local_model}")
                self._local_model_instance = TextEmbedding(
                    model_name=self.local_model,
                    cache_dir=self.fastembed_cache_dir
                )
                logger.info("Local embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {str(e)}")
                raise Exception(f"Local embedding model initialization failed: {str(e)}") from e
        
        return self._local_model_instance
    
    def _generate_remote_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenRouter API."""
        try:
            response = requests.post(
                f"{self.openrouter_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.openrouter_model,
                    "input": text,
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            embedding = data["data"][0]["embedding"]
            
            logger.info(f"Remote embedding generated (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Remote embedding failed: {str(e)}")
            raise
    
    def _generate_remote_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenRouter API (batch)."""
        try:
            response = requests.post(
                f"{self.openrouter_base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.openrouter_model,
                    "input": texts,
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            
            logger.info(f"Remote batch embeddings generated ({len(embeddings)} embeddings)")
            return embeddings
            
        except Exception as e:
            logger.error(f"Remote batch embedding failed: {str(e)}")
            raise
    
    def _generate_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local FastEmbed model."""
        try:
            model = self._get_local_model()
            embeddings = list(model.embed([text]))
            embedding = embeddings[0].tolist()
            
            logger.info(f"Local embedding generated (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Local embedding failed: {str(e)}")
            raise
    
    def _generate_local_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local FastEmbed model (batch)."""
        try:
            model = self._get_local_model()
            embeddings = list(model.embed(texts))
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            logger.info(f"Local batch embeddings generated ({len(embeddings_list)} embeddings)")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Local batch embedding failed: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with auto-fallback.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If both remote and local embedding fail
        """
        # Try current method
        try:
            if self.current_method == "remote":
                logger.info("Attempting remote embedding...")
                return self._generate_remote_embedding(text)
            else:
                logger.info("Attempting local embedding...")
                return self._generate_local_embedding(text)
        except Exception as e:
            logger.warning(f"Primary embedding method ({self.current_method}) failed: {str(e)}")
            
            # Try fallback if enabled
            if self.auto_fallback and self.current_method == "remote":
                logger.info("Auto-fallback enabled, switching to local embedding...")
                try:
                    embedding = self._generate_local_embedding(text)
                    logger.info("✅ Fallback to local embedding successful")
                    # Update current method for future calls
                    self.current_method = "local"
                    return embedding
                except Exception as fallback_error:
                    logger.error(f"Fallback to local embedding also failed: {str(fallback_error)}")
                    raise Exception(f"Both remote and local embedding failed. Remote: {str(e)}, Local: {str(fallback_error)}") from fallback_error
            else:
                raise Exception(f"Embedding generation failed: {str(e)}") from e
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with auto-fallback.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If both remote and local embedding fail
        """
        # Try current method
        try:
            if self.current_method == "remote":
                logger.info(f"Attempting remote batch embedding for {len(texts)} texts...")
                return self._generate_remote_embeddings_batch(texts)
            else:
                logger.info(f"Attempting local batch embedding for {len(texts)} texts...")
                return self._generate_local_embeddings_batch(texts)
        except Exception as e:
            logger.warning(f"Primary embedding method ({self.current_method}) failed: {str(e)}")
            
            # Try fallback if enabled
            if self.auto_fallback and self.current_method == "remote":
                logger.info("Auto-fallback enabled, switching to local embedding...")
                try:
                    embeddings = self._generate_local_embeddings_batch(texts)
                    logger.info("✅ Fallback to local embedding successful")
                    # Update current method for future calls
                    self.current_method = "local"
                    return embeddings
                except Exception as fallback_error:
                    logger.error(f"Fallback to local embedding also failed: {str(fallback_error)}")
                    raise Exception(f"Both remote and local embedding failed. Remote: {str(e)}, Local: {str(fallback_error)}") from fallback_error
            else:
                raise Exception(f"Batch embedding generation failed: {str(e)}") from e
    
    def switch_method(self, method: str) -> str:
        """
        Switch embedding method.
        
        Args:
            method: "remote" or "local"
            
        Returns:
            Confirmation message
        """
        if method not in ["remote", "local"]:
            raise ValueError("Method must be 'remote' or 'local'")
        
        old_method = self.current_method
        self.current_method = method
        logger.info(f"Embedding method switched: {old_method} → {method}")
        
        return f"Embedding method switched from '{old_method}' to '{method}'"
    
    def get_current_method(self) -> str:
        """Get current embedding method."""
        return self.current_method
    
    def get_dimension(self) -> int:
        """Get dimension for current method."""
        if self.current_method == "remote":
            return self.remote_dimension
        else:
            return self.local_dimension


# Global embedding client instance
_embedding_client = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create the global embedding client instance."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
