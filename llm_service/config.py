import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator, root_validator

from .exceptions import LLMConfigError

# --- Pydantic Models for Configuration Structure ---

class LLMModelSettings(BaseModel):
    """Default settings for a specific LLM model."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    # Ollama specific
    num_ctx: Optional[int] = None
    keep_alive: Optional[str] = None # e.g., "5m", "-1" (keep alive indefinitely)
    # Add other common settings as needed, or allow arbitrary through extra='allow'

    class Config:
        extra = 'allow' # Allow other provider-specific settings

class LLMModelConfig(BaseModel):
    """Configuration for a specific LLM model."""
    model_name: str # Actual model identifier for the provider (e.g., "gpt-3.5-turbo", "llama3:8b")
    default_settings: Optional[LLMModelSettings] = Field(default_factory=LLMModelSettings)
    # Allow other model-specific parameters not covered above

    class Config:
        extra = 'allow'

class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider (e.g., OpenAI, Ollama)."""
    api_key_env_var: Optional[str] = None # Name of the environment variable for the API key
    api_key: Optional[str] = None # Actual API key (populated from env_var or directly)
    base_url: Optional[str] = None # e.g., for Ollama or custom OpenAI endpoints
    default_keep_alive: Optional[str] = None # Specific to Ollama provider
    models: Dict[str, LLMModelConfig]

    @root_validator(pre=True)
    def load_api_key_from_env(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Load API key from environment variable if specified and not already set."""
        api_key = values.get("api_key")
        api_key_env_var = values.get("api_key_env_var")

        if not api_key and api_key_env_var:
            try:
                values["api_key"] = os.environ[api_key_env_var]
            except KeyError:
                # Warning or error can be raised here if key is strictly required upon load
                # For now, we allow it to be None, specific service will handle missing key
                pass
        return values

    class Config:
        extra = 'allow'


class LLMProvidersRoot(BaseModel):
    """Root structure for all LLM provider configurations."""
    openai: Optional[LLMProviderConfig] = None
    ollama: Optional[LLMProviderConfig] = None
    # Add other providers here as they are supported, e.g.
    # anthropic: Optional[LLMProviderConfig] = None

    class Config:
        extra = 'allow' # Allow other top-level provider keys

class LLMConfigFile(BaseModel):
    """Represents the overall structure of the llm_config.yaml file."""
    default_model_id: Optional[str] = None
    llm_providers: LLMProvidersRoot
    global_timeout: Optional[int] = Field(default=60, description="Global timeout in seconds for LLM requests")


class LLMConfigManager:
    """
    Handles loading, parsing, and providing access to LLM configurations
    from a YAML file.
    """
    _config: Optional[LLMConfigFile] = None
    _config_file_path: Optional[Path] = None

    def __init__(self, config_file_path: Optional[str] = None):
        """
        Initializes the LLMConfigManager.

        Args:
            config_file_path: Path to the llm_config.yaml file.
                              If None, defaults to "llm_config.yaml" in the same directory as this file.
        """
        if config_file_path:
            self._config_file_path = Path(config_file_path)
        else:
            self._config_file_path = Path(__file__).parent / "llm_config.yaml"

        self.load_configuration()

    def load_configuration(self):
        """
        Loads the LLM configuration from the YAML file.
        Populates self._config with the parsed and validated data.

        Raises:
            LLMConfigError: If the config file is not found, cannot be parsed,
                            or fails Pydantic validation.
        """
        if not self._config_file_path.is_file():
            raise LLMConfigError(f"Configuration file not found at: {self._config_file_path}")

        try:
            with open(self._config_file_path, 'r') as f:
                raw_config = yaml.safe_load(f)
            if not raw_config:
                 raise LLMConfigError(f"Configuration file is empty or invalid: {self._config_file_path}")

            self._config = LLMConfigFile(**raw_config)

        except yaml.YAMLError as e:
            raise LLMConfigError(f"Error parsing YAML configuration file: {self._config_file_path}", details=str(e))
        except Exception as e: # Catches Pydantic ValidationError and other unexpected errors
            raise LLMConfigError(f"Error validating configuration from {self._config_file_path}", details=str(e))

    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """
        Retrieves the configuration for a specific LLM provider.

        Args:
            provider_name: The name of the provider (e.g., "openai", "ollama").

        Returns:
            The LLMProviderConfig for the provider, or None if not found.
        """
        if not self._config:
            self.load_configuration() # Ensure config is loaded

        if self._config and self._config.llm_providers:
            return getattr(self._config.llm_providers, provider_name, None)
        return None

    def get_model_config(self, provider_name: str, model_key: str) -> Optional[LLMModelConfig]:
        """
        Retrieves the configuration for a specific model under a provider.

        Args:
            provider_name: The name of the provider.
            model_key: The key/alias of the model within the provider's config
                       (e.g., "gpt_3_5_turbo", "llama3_8b").

        Returns:
            The LLMModelConfig for the model, or None if not found.
        """
        provider_config = self.get_provider_config(provider_name)
        if provider_config and provider_config.models:
            return provider_config.models.get(model_key)
        return None

    def get_default_model_id(self) -> Optional[str]:
        """Returns the default_model_id specified in the config."""
        if not self._config:
            self.load_configuration()
        return self._config.default_model_id if self._config else None

    def get_default_model_config(self) -> Optional[LLMModelConfig]:
        """
        Retrieves the configuration for the default_model_id.
        The default_model_id is expected to be in the format "provider_name_model_key".
        """
        default_model_id = self.get_default_model_id()
        if not default_model_id:
            return None

        # Attempt to parse provider and model_key from default_model_id
        # This is a simple convention; more robust parsing might be needed.
        parts = default_model_id.split('_', 1)
        if len(parts) != 2:
            # Or log a warning and return None
            raise LLMConfigError(f"Invalid default_model_id format: '{default_model_id}'. Expected 'provider_modelkey'.")

        provider_name, model_key_candidate = parts[0], parts[1]

        # Need to find the actual model_key if default_model_id is an alias like "openai_gpt_3_5_turbo"
        # and the key in YAML is "gpt_3_5_turbo"
        provider_config = self.get_provider_config(provider_name)
        if provider_config and provider_config.models:
            # Try direct match first (if model_key_candidate is the actual key)
            if model_key_candidate in provider_config.models:
                 return provider_config.models[model_key_candidate]
            # Try to find a model_key that, when provider_name is prepended, matches default_model_id
            for key, model_cfg in provider_config.models.items():
                if f"{provider_name}_{key}" == default_model_id:
                    return model_cfg

        raise LLMConfigError(f"Default model configuration for ID '{default_model_id}' not found.")


    def get_global_timeout(self) -> Optional[int]:
        if not self._config:
            self.load_configuration()
        return self._config.global_timeout if self._config else 60

    def get_all_providers_config(self) -> Dict[str, LLMProviderConfig]:
        """Returns the raw configuration for all providers."""
        if self._config and self._config.llm_providers:
            # Return a dict of the provider Pydantic model instances
            providers = {}
            if self._config.llm_providers.openai:
                providers["openai"] = self._config.llm_providers.openai
            if self._config.llm_providers.ollama:
                providers["ollama"] = self._config.llm_providers.ollama
            # Add other providers if they become available
            return providers
        return {}


if __name__ == "__main__":
    # Example Usage (assuming llm_config.yaml is in the same directory or specified path)
    try:
        # Create a dummy llm_config.yaml for testing this main block
        dummy_config_content = """
default_model_id: "openai_gpt_3_5_turbo"
llm_providers:
  openai:
    api_key_env_var: "OPENAI_API_KEY_TEST" # Use a test env var
    models:
      gpt_3_5_turbo:
        model_name: "gpt-3.5-turbo"
        default_settings:
          temperature: 0.8
      gpt_4o:
        model_name: "gpt-4o"
  ollama:
    base_url: "http://localhost:11434"
    models:
      llama3:
        model_name: "llama3"
        default_settings:
          num_ctx: 2048
"""
        dummy_config_path = Path(__file__).parent / "llm_config.yaml"
        with open(dummy_config_path, 'w') as f:
            f.write(dummy_config_content)

        # Set a dummy env var for testing
        os.environ["OPENAI_API_KEY_TEST"] = "test_api_key_from_env"

        manager = LLMConfigManager() # Uses llm_config.yaml in the same dir

        print(f"Default Model ID: {manager.get_default_model_id()}")

        openai_cfg = manager.get_provider_config("openai")
        if openai_cfg:
            print(f"\nOpenAI Config API Key: {openai_cfg.api_key}")
            gpt3_cfg = manager.get_model_config("openai", "gpt_3_5_turbo")
            if gpt3_cfg:
                print(f"GPT-3.5 Turbo Model Name: {gpt3_cfg.model_name}")
                print(f"GPT-3.5 Turbo Default Temp: {gpt3_cfg.default_settings.temperature}")

        default_model_cfg = manager.get_default_model_config()
        if default_model_cfg:
            print(f"\nDefault Model ({manager.get_default_model_id()}) Config:")
            print(f"  Actual Model Name: {default_model_cfg.model_name}")
            print(f"  Default Settings: {default_model_cfg.default_settings.model_dump_json(indent=2)}")

        ollama_cfg = manager.get_provider_config("ollama")
        if ollama_cfg:
            llama3_cfg = manager.get_model_config("ollama", "llama3")
            if llama3_cfg:
                 print(f"\nOllama Llama3 Model Name: {llama3_cfg.model_name}")
                 print(f"Ollama Llama3 Default num_ctx: {llama3_cfg.default_settings.num_ctx}")

        print(f"\nGlobal Timeout: {manager.get_global_timeout()}")

        # Clean up dummy file and env var
        dummy_config_path.unlink(missing_ok=True)
        del os.environ["OPENAI_API_KEY_TEST"]

    except LLMConfigError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
