"""
Configuration settings for Zilliz MCP Server.

Simple configuration module that loads Zilliz settings from .env file.
"""

import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file only if it exists in current directory
# This ensures uvx environment variables take precedence
import os
if os.path.exists('.env'):
    load_dotenv(override=False)  # Don't override existing environment variables


class ZillizConfig:
    """Zilliz Cloud configuration."""
    
    def __init__(self):
        """Initialize configuration with validation."""
        self.cloud_uri: str = os.getenv("ZILLIZ_CLOUD_URI", "https://api.cloud.zilliz.com")
        self.token: str = os.getenv("ZILLIZ_CLOUD_TOKEN", "")
        self.free_cluster_region: str = os.getenv("ZILLIZ_CLOUD_FREE_CLUSTER_REGION", "gcp-us-west1")
        
        # MCP Server configuration
        try:
            self.mcp_server_port: int = int(os.getenv("MCP_SERVER_PORT", "8000"))
        except ValueError:
            raise ValueError("MCP_SERVER_PORT must be a valid integer")
            
        self.mcp_server_host: str = os.getenv("MCP_SERVER_HOST", "localhost")
        
        # Embedding configuration
        self.enable_auto_embedding: bool = os.getenv("ENABLE_AUTO_EMBEDDING", "false").lower() == "true"
        self.openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
        self.openrouter_embedding_model: str = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
        try:
            self.embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
        except ValueError:
            raise ValueError("EMBEDDING_DIMENSION must be a valid integer")
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration values."""
        # Only validate token as strictly required - other configs have reasonable defaults
        if not self.token or not self.token.strip():
            raise ValueError("ZILLIZ_CLOUD_TOKEN is required and cannot be empty. Please set your Zilliz Cloud API token.")
        
        # Validate cloud URI format only if it's not the default
        if self.cloud_uri and not re.match(r'^https?://', self.cloud_uri):
            raise ValueError("ZILLIZ_CLOUD_URI must be a valid URL starting with http:// or https://")
    

        # Validate MCP server port range
        if not (1 <= self.mcp_server_port <= 65535):
            raise ValueError("MCP_SERVER_PORT must be between 1 and 65535")
        
        # Validate embedding configuration if auto-embedding is enabled
        if self.enable_auto_embedding:
            if not self.openrouter_api_key or not self.openrouter_api_key.strip():
                raise ValueError("OPENROUTER_API_KEY is required when ENABLE_AUTO_EMBEDDING is true")
            if self.embedding_dimension <= 0:
                raise ValueError("EMBEDDING_DIMENSION must be a positive integer")


# Global config instance
config = ZillizConfig()


def get_config() -> ZillizConfig:
    """Get the Zilliz configuration."""
    return config

