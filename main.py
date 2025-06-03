import logging
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status
from typing import Any, Dict, List, Optional

# Assuming llm_service is in the Python path
from llm_service import (
    LLMConfigManager,
    LLMManager,
    PromptTemplateManager,
    LLMModelNotFoundError,
    PromptTemplateNotFoundError,
    LLMServiceError, # Catchall for other LLM issues
    LLMContext
)

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Application Setup ---
app = FastAPI(
    title="LLM Service Integration API",
    description="An API to interact with various LLM services and manage prompts.",
    version="0.1.0"
)

# --- Configuration Paths ---
# Define paths relative to this main.py file or use environment variables for flexibility
# For this example, we assume main.py is at the project root.
PROJECT_ROOT = Path(__file__).parent.resolve()
LLM_CONFIG_PATH = PROJECT_ROOT / "llm_service" / "llm_config.yaml"
# PromptTemplateManager defaults to llm_service/prompts/prompt_templates.yaml,
# which is fine if template_manager.py is in llm_service/prompts/
# If PromptTemplateManager was in llm_service/ itself, we might need:
# PROMPT_TEMPLATES_PATH = PROJECT_ROOT / "llm_service" / "prompts" / "prompt_templates.yaml"


# --- Application State and Lifecycle ---

@app.on_event("startup")
async def startup_event():
    """
    Initialize LLM managers and configurations at application startup.
    """
    logger.info(f"Loading LLM configuration from: {LLM_CONFIG_PATH}")
    if not LLM_CONFIG_PATH.is_file():
        logger.error(f"LLM Configuration file not found at {LLM_CONFIG_PATH}")
        # This is a critical error, app might not function.
        # Depending on requirements, could raise an exception to stop startup,
        # or allow it to start in a degraded state.
        # For now, we'll log and services relying on it will fail.
        app.state.llm_config_manager = None
        app.state.llm_manager = None
        app.state.prompt_template_manager = None
        return

    try:
        app.state.llm_config_manager = LLMConfigManager(config_file_path=str(LLM_CONFIG_PATH))
        logger.info("LLMConfigManager initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize LLMConfigManager: {e}", exc_info=True)
        app.state.llm_config_manager = None # Ensure it's None if init fails
        # Propagate other managers to None as well since they depend on config_manager
        app.state.llm_manager = None
        app.state.prompt_template_manager = None
        return


    try:
        # PromptTemplateManager uses a default path if None is provided,
        # which is llm_service/prompts/prompt_templates.yaml relative to template_manager.py
        app.state.prompt_template_manager = PromptTemplateManager()
        logger.info("PromptTemplateManager initialized successfully (using default path).")
    except Exception as e:
        logger.error(f"Failed to initialize PromptTemplateManager: {e}", exc_info=True)
        app.state.prompt_template_manager = None


    if app.state.llm_config_manager:
        try:
            app.state.llm_manager = LLMManager(config_manager=app.state.llm_config_manager)
            logger.info("LLMManager initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMManager: {e}", exc_info=True)
            app.state.llm_manager = None
    else:
        logger.warning("LLMManager could not be initialized because LLMConfigManager failed to load.")
        app.state.llm_manager = None


@app.on_event("shutdown")
async def shutdown_event():
    """
    Gracefully close LLM services on application shutdown.
    """
    if hasattr(app.state, 'llm_manager') and app.state.llm_manager:
        logger.info("Closing LLM services...")
        await app.state.llm_manager.close_all_services()
        logger.info("LLM services closed.")
    else:
        logger.info("No LLMManager found in app state to close services.")

# --- Dependencies ---

def get_llm_manager() -> LLMManager:
    """
    FastAPI dependency to get the LLMManager instance.
    Raises HTTPException if the manager is not available.
    """
    if not hasattr(app.state, 'llm_manager') or not app.state.llm_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLMManager is not available or not initialized correctly."
        )
    return app.state.llm_manager

def get_prompt_template_manager() -> PromptTemplateManager:
    """
    FastAPI dependency to get the PromptTemplateManager instance.
    Raises HTTPException if the manager is not available.
    """
    if not hasattr(app.state, 'prompt_template_manager') or not app.state.prompt_template_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PromptTemplateManager is not available or not initialized correctly."
        )
    return app.state.prompt_template_manager


# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the LLM Service Integration API. Visit /docs for API documentation."}

# Routers will be included here
from llm_service.api_v1_llm import router as llm_api_router
app.include_router(llm_api_router, prefix="/api/v1", tags=["LLM Operations"]) # Using /api/v1 as prefix for the router

logger.info("FastAPI application initialized with LLM router.")
