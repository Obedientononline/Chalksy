import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OMI_WEBHOOK_SECRET: str = "omi_hackathon_secret_2026"
    
    # Qdrant Memory Configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_STORAGE_PATH: str = "./data/qdrant_storage"
    
    # AI Providers
    LYZR_API_KEY: Optional[str] = None
    LYZR_AGENT_ID: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Omi App Integration credentials (for sending notifications)
    OMI_APP_ID: Optional[str] = None
    OMI_APP_SECRET: Optional[str] = None
    # Execution Flags
    OFFLINE_MODE: bool = False
    LOCAL_ENCRYPTION_KEY: str = "super_secure_32_byte_aes_secret_key_hackathon!"
    STUDENT_MAPPING_DB: str = "./data/student_registry.db"
    TASK_STATE_DB: str = "./data/tasks.db"
    
    # Host & Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
