import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from llm_service.config import (
    LLMConfigManager,
    LLMConfigError,
    LLMConfigFile,
    LLMProviderConfig,
    LLMModelConfig,
    LLMModelSettings
)
from pydantic import ValidationError

# --- Fixtures ---

@pytest.fixture
def valid_config_yaml_content() -> str:
    return """
default_model_id: "openai_gpt_3_5_turbo"
global_timeout: 90 # Changed from global_request_timeout to match Pydantic model
llm_providers:
  openai:
    provider_name: "openai"
    api_key_env_var: "TEST_OPENAI_API_KEY"
    default_keep_alive: "10m" # Example of a provider-level default
    models:
      gpt_3_5_turbo:
        model_name: "gpt-3.5-turbo-0125"
        default_settings:
          temperature: 0.7
          max_tokens: 1000
      gpt_4_o:
        model_name: "gpt-4o"
        # No default_settings, should inherit from provider or use LLMModelSettings defaults
  ollama:
    provider_name: "ollama"
    base_url: "http://localhost:11434"
    default_keep_alive: "1h"
    models:
      llama3_8b:
        model_name: "llama3:8b-instruct"
        default_settings:
          temperature: 0.5
          num_ctx: 4096
          keep_alive: "30m" # Model-specific keep_alive overrides provider
"""

@pytest.fixture
def minimal_config_yaml_content() -> str:
    return """
llm_providers:
  openai:
    models:
      test_model:
        model_name: "test-model-name"
"""

@pytest.fixture
def config_file(tmp_path: Path, request) -> Path:
    """
    Creates a temporary config file.
    The content for the file is determined by request.param.
    If request.param is a string, it's treated as a fixture name to resolve.
    Otherwise, request.param is used directly as content.
    If request.param is not set, it defaults to the content from 'valid_config_yaml_content' fixture.
    """
    content_source = getattr(request, "param", "valid_config_yaml_content")

    if isinstance(content_source, str):
        # If it's a string, assume it's a fixture name and resolve it
        content = request.getfixturevalue(content_source)
    else:
        # Otherwise, use the param directly as content (e.g., if not using indirect parametrization)
        # This case might not be hit with current test setup but makes fixture more robust.
        content = content_source

    p = tmp_path / "llm_config.yaml"
    p.write_text(content)
    return p

# --- Test Cases ---

class TestLLMConfigManagerSuccess:

    @patch.dict(os.environ, {"TEST_OPENAI_API_KEY": "fake_openai_key"})
    def test_successful_loading(self, config_file: Path):
        manager = LLMConfigManager(config_file_path=str(config_file))

        assert manager.get_default_model_id() == "openai_gpt_3_5_turbo"
        assert manager.get_global_timeout() == 90 # Corrected method name

        # Test OpenAI provider
        openai_provider = manager.get_provider_config("openai")
        assert openai_provider is not None
        assert openai_provider.provider_name == "openai"
        assert openai_provider.api_key == "fake_openai_key" # Check resolved API key
        assert openai_provider.default_keep_alive == "10m"

        # Test OpenAI models
        gpt3_model = manager.get_model_config("openai", "gpt_3_5_turbo")
        assert gpt3_model is not None
        assert gpt3_model.model_name == "gpt-3.5-turbo-0125"
        assert gpt3_model.default_settings.temperature == 0.7
        assert gpt3_model.default_settings.max_tokens == 1000
        # keep_alive should be from model_settings if defined, else provider, else default.
        # gpt3_model has no keep_alive, so it should inherit openai_provider's "10m"
        # However, the current model structure means LLMModelSettings will have its own default if not set.
        # Let's check the effective keep_alive.
        # The Pydantic models in config.py will set default keep_alive if not specified.
        # LLMModelSettings has `keep_alive: Optional[str] = None`
        # LLMProviderConfig has `default_keep_alive: Optional[str] = None`
        # The services (e.g. OllamaService) merge these. Config manager just loads.
        assert gpt3_model.default_settings.keep_alive is None # Not set at model level

        gpt4_model = manager.get_model_config("openai", "gpt_4_o")
        assert gpt4_model is not None
        assert gpt4_model.model_name == "gpt-4o"
        assert gpt4_model.default_settings is not None # Should have default LLMModelSettings
        assert gpt4_model.default_settings.temperature is None # Pydantic default for LLMModelSettings

        # Test Ollama provider
        ollama_provider = manager.get_provider_config("ollama")
        assert ollama_provider is not None
        assert ollama_provider.provider_name == "ollama"
        assert ollama_provider.base_url == "http://localhost:11434"
        assert ollama_provider.default_keep_alive == "1h"

        # Test Ollama models
        llama3_model = manager.get_model_config("ollama", "llama3_8b")
        assert llama3_model is not None
        assert llama3_model.model_name == "llama3:8b-instruct"
        assert llama3_model.default_settings.temperature == 0.5
        assert llama3_model.default_settings.num_ctx == 4096
        assert llama3_model.default_settings.keep_alive == "30m" # Model specific

        # Test get_default_model_config
        default_model_config = manager.get_default_model_config()
        assert default_model_config is not None
        assert default_model_config.model_name == "gpt-3.5-turbo-0125"

        # Test get_all_providers_config
        all_providers = manager.get_all_providers_config()
        assert "openai" in all_providers
        assert "ollama" in all_providers
        assert len(all_providers["openai"].models) == 2
        assert len(all_providers["ollama"].models) == 1

    @pytest.mark.parametrize("config_file", ["minimal_config_yaml_content"], indirect=True)
    def test_minimal_config_loading(self, config_file: Path):
        """Test loading with a minimal valid configuration."""
        manager = LLMConfigManager(config_file_path=str(config_file))
        assert manager.get_default_model_id() is None # No default_model_id specified
        assert manager.get_global_timeout() == 60 # Expect Pydantic default

        openai_provider = manager.get_provider_config("openai")
        assert openai_provider is not None
        assert openai_provider.api_key_env_var is None # Not specified in minimal
        assert openai_provider.api_key is None

        test_model = manager.get_model_config("openai", "test_model")
        assert test_model is not None
        assert test_model.model_name == "test-model-name"
        assert test_model.default_settings is not None # Default LLMModelSettings


class TestLLMConfigManagerErrorHandling:

    def test_file_not_found(self):
        with pytest.raises(LLMConfigError, match="Configuration file not found"):
            LLMConfigManager(config_file_path="/tmp/non_existent_config.yaml")

    @patch("builtins.open", new_callable=mock_open, read_data="invalid_yaml: [")
    def test_invalid_yaml_syntax(self, mock_file, tmp_path: Path):
        # tmp_path is used to give a real path to ConfigManager, but open is mocked
        dummy_path = tmp_path / "dummy.yaml"
        dummy_path.touch() # Ensure the file exists for is_file() check
        with pytest.raises(LLMConfigError, match="Error parsing YAML configuration"):
            LLMConfigManager(config_file_path=str(dummy_path))

    @patch.dict(os.environ, {}, clear=True) # Ensure no env var is set
    def test_api_key_env_var_not_set(self, tmp_path: Path):
        config_content = """
llm_providers:
  openai:
    api_key_env_var: "MISSING_OPENAI_KEY"
    models:
      gpt_3_5_turbo:
        model_name: "gpt-3.5-turbo"
"""
        config_file = tmp_path / "api_key_missing.yaml"
        config_file.write_text(config_content)

        # Validator in LLMProviderConfig currently does not raise error if env var is not set,
        # but logs a warning and api_key remains None.
        # So, LLMConfigManager should load without error here.
        manager = LLMConfigManager(config_file_path=str(config_file))
        openai_provider = manager.get_provider_config("openai")
        assert openai_provider is not None
        assert openai_provider.api_key is None # Key should be None as env var was not set

    def test_pydantic_validation_error_missing_providers(self, tmp_path: Path):
        config_content = "default_model_id: 'some_model'" # llm_providers is missing
        config_file = tmp_path / "missing_providers.yaml"
        config_file.write_text(config_content)
        expected_error_msg_regex = f"Error validating configuration from {config_file}"
        with pytest.raises(LLMConfigError, match=expected_error_msg_regex):
            LLMConfigManager(config_file_path=str(config_file))

    def test_pydantic_validation_error_missing_model_name(self, tmp_path: Path):
        config_content = """
llm_providers:
  openai:
    models:
      my_model: {} # model_name is missing
"""
        config_file = tmp_path / "missing_model_name.yaml"
        config_file.write_text(config_content)
        expected_error_msg_regex = f"Error validating configuration from {config_file}"
        with pytest.raises(LLMConfigError, match=expected_error_msg_regex):
            LLMConfigManager(config_file_path=str(config_file))

    def test_pydantic_validation_error_wrong_type(self, tmp_path: Path):
        config_content = """
llm_providers:
  openai:
    models:
      my_model:
        model_name: "test"
        default_settings:
          temperature: "should_be_float"
"""
        config_file = tmp_path / "wrong_type.yaml"
        config_file.write_text(config_content)
        expected_error_msg_regex = f"Error validating configuration from {config_file}"
        with pytest.raises(LLMConfigError, match=expected_error_msg_regex):
            LLMConfigManager(config_file_path=str(config_file))


class TestLLMConfigManagerEdgeCases:

    def test_empty_yaml_file(self, tmp_path: Path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("") # PyYAML loads this as None
        # LLMConfigManager raises "Configuration file is empty or invalid" before Pydantic validation
        expected_error_msg_regex = f"Configuration file is empty or invalid: {config_file}"
        with pytest.raises(LLMConfigError, match=expected_error_msg_regex):
            LLMConfigManager(config_file_path=str(config_file))

    def test_no_providers_in_config(self, tmp_path: Path):
        config_content = "llm_providers: {}"
        config_file = tmp_path / "no_providers.yaml"
        config_file.write_text(config_content)
        manager = LLMConfigManager(config_file_path=str(config_file))
        # This test will be fixed after adding get_all_providers_config to the manager
        # For now, let's assume the method will exist:
        assert manager.get_all_providers_config() == {}
        assert manager.get_default_model_id() is None

    def test_provider_with_no_models(self, tmp_path: Path):
        config_content = """
llm_providers:
  openai:
    models: {}
"""
        config_file = tmp_path / "no_models.yaml"
        config_file.write_text(config_content)
        manager = LLMConfigManager(config_file_path=str(config_file))
        openai_provider = manager.get_provider_config("openai")
        assert openai_provider is not None
        assert openai_provider.models == {}

    def test_default_model_id_not_found(self, tmp_path: Path):
        config_content = """
default_model_id: "non_existent_provider_non_existent_model"
llm_providers:
  openai:
    models:
      gpt_3_5_turbo:
        model_name: "gpt-3.5-turbo-0125"
"""
        config_file = tmp_path / "default_not_found.yaml"
        config_file.write_text(config_content)
        manager = LLMConfigManager(config_file_path=str(config_file))
        # Default model ID is stored as is, but get_default_model_config will fail
        assert manager.get_default_model_id() == "non_existent_provider_non_existent_model"
        expected_error_msg_regex = "Default model configuration for ID 'non_existent_provider_non_existent_model' not found."
        with pytest.raises(LLMConfigError, match=expected_error_msg_regex):
            manager.get_default_model_config()

    def test_get_model_config_provider_not_found(self, config_file: Path):
        manager = LLMConfigManager(config_file_path=str(config_file))
        assert manager.get_model_config("unknown_provider", "some_model") is None

    def test_get_model_config_model_not_found_in_provider(self, config_file: Path):
        manager = LLMConfigManager(config_file_path=str(config_file))
        assert manager.get_model_config("openai", "unknown_model") is None
