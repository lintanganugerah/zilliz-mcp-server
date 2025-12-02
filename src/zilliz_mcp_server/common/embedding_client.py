"""OpenRouter embedding client for auto-generating embeddings."""

import logging
import requests
from typing import List, Union
from zilliz_mcp_server.settings import config

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Client for generating embeddings via OpenRouter API."""
    
    def __init__(self):
        """Initialize embedding client."""
        self.api_key = config.openrouter_api_key
        self.model = config.openrouter_embedding_model
        self.dimension = config.embedding_dimension
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.info(f"Generating embedding for text (length: {len(text)})")
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": text,
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embedding from response
            embedding = data["data"][0]["embedding"]
            
            logger.info(f"Successfully generated embedding (dimension: {len(embedding)})")
            return embedding
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}") from e
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from OpenRouter: {str(e)}")
            raise Exception(f"Invalid embedding response format: {str(e)}") from e
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=60
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Batch embedding generation failed: {str(e)}") from e
        except (KeyError, IndexError) as e:
            logger.error(f"Invalid response format from OpenRouter: {str(e)}")
            raise Exception(f"Invalid embedding response format: {str(e)}") from e


# Global embedding client instance
_embedding_client = None


def get_embedding_client() -> EmbeddingClient:
    """Get or create the global embedding client instance."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client
