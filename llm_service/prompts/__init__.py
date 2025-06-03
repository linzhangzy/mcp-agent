# This file makes 'prompts' a sub-package of 'llm_service'.

from .template_manager import PromptTemplateManager

__all__ = [
    "PromptTemplateManager",
]
