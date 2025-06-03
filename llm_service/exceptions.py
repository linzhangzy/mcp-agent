class LLMServiceError(Exception):
    """Base class for exceptions in the LLM service module."""
    pass

class LLMConfigError(LLMServiceError):
    """Exceptions related to LLM configuration loading or validation."""
    def __init__(self, message: str, details: str = None):
        super().__init__(message)
        self.details = details

    def __str__(self):
        if self.details:
            return f"{super().__str__()} - Details: {self.details}"
        return super().__str__()

class LLMResponseError(LLMServiceError):
    """Exceptions related to errors received from the LLM provider or during response processing."""
    def __init__(self, message: str, provider_name: str = None, status_code: int = None):
        super().__init__(message)
        self.provider_name = provider_name
        self.status_code = status_code

    def __str__(self):
        parts = [super().__str__()]
        if self.provider_name:
            parts.append(f"Provider: {self.provider_name}")
        if self.status_code is not None:
            parts.append(f"Status Code: {self.status_code}")
        return " - ".join(parts)

class LLMPluginError(LLMServiceError):
    """Exceptions specific to LLM plugin loading or execution."""
    pass

class LLMModelNotFoundError(LLMServiceError):
    """Exception raised when a specific LLM model ID is not found or could not be loaded."""
    def __init__(self, model_id: str, message: str = None):
        self.model_id = model_id
        if message is None:
            message = f"LLM model with ID '{model_id}' not found or could not be loaded."
        super().__init__(message)

    def __str__(self):
        return f"{super().__str__()} (Model ID: {self.model_id})"


class PromptTemplateError(LLMServiceError):
    """Base class for exceptions related to prompt templates."""
    pass


class PromptTemplateNotFoundError(PromptTemplateError):
    """Exception raised when a specific prompt template is not found."""
    def __init__(self, template_name: str, message: str = None):
        self.template_name = template_name
        if message is None:
            message = f"Prompt template '{template_name}' not found."
        super().__init__(message)

    def __str__(self):
        return f"{super().__str__()} (Template Name: {self.template_name})"
