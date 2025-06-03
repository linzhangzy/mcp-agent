# LLM Service Module
# This module provides interfaces and utilities for integrating various
# Large Language Model services.

from .base import AbstractLLMService, LLMContext
from .config import (
    LLMConfigManager,
    LLMProviderConfig,
    LLMModelConfig,
    LLMModelSettings,
    LLMProvidersRoot, # Added for completeness if someone wants to type the whole provider block
    LLMConfigFile # Added for completeness
)
from .exceptions import (
    LLMServiceError,
    LLMConfigError,
    LLMResponseError,
    LLMPluginError,
    LLMModelNotFoundError,
    PromptTemplateError, # Added PromptTemplateError
    PromptTemplateNotFoundError # Added PromptTemplateNotFoundError
)
from .openai_service import OpenAIService
from .ollama_service import OllamaService
from .manager import LLMManager
from .prompts import PromptTemplateManager # Added PromptTemplateManager

__all__ = [
    # From base.py
    "AbstractLLMService",
    "LLMContext",
    # From config.py
    "LLMConfigManager",
    "LLMProviderConfig",
    "LLMModelConfig",
    "LLMModelSettings",
    "LLMProvidersRoot",
    "LLMConfigFile",
    # From exceptions.py
    "LLMServiceError",
    "LLMConfigError",
    "LLMResponseError",
    "LLMPluginError",
    "LLMModelNotFoundError",
    "PromptTemplateError", # Added PromptTemplateError
    "PromptTemplateNotFoundError", # Added PromptTemplateNotFoundError
    # From openai_service.py
    "OpenAIService",
    # From ollama_service.py
    "OllamaService",
    # From manager.py
    "LLMManager",
    # From prompts sub-package
    "PromptTemplateManager", # Added PromptTemplateManager
]
