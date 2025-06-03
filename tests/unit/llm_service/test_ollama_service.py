import pytest
import httpx
import os
import json # For formatting error details in tests
from unittest.mock import patch, AsyncMock, MagicMock

from llm_service.ollama_service import OllamaService, DEFAULT_OLLAMA_BASE_URL, DEFAULT_TIMEOUT as OLLAMA_DEFAULT_TIMEOUT
from llm_service.base import LLMContext
from llm_service.config import LLMConfigManager, LLMProviderConfig, LLMModelConfig, LLMModelSettings
from llm_service.exceptions import LLMConfigError, LLMResponseError

# --- Fixtures ---

@pytest.fixture
def mock_llm_config_manager():
    """Provides a MagicMock for LLMConfigManager."""
    mock_manager = MagicMock(spec=LLMConfigManager)
    # Setup default return for global timeout if not overridden
    mock_manager.get_global_timeout.return_value = None
    return mock_manager

@pytest.fixture
def sample_ollama_provider_config():
    return LLMProviderConfig(
        provider_name="ollama",
        base_url="http://mocked-ollama:11434",
        default_keep_alive="10m",
        models={}
    )

@pytest.fixture
def sample_llama2_model_config():
    return LLMModelConfig(
        model_name="llama2:latest", # This is the Ollama model tag
        default_settings=LLMModelSettings(temperature=0.6, num_ctx=2048, keep_alive="5m")
    )

@pytest.fixture
def ollama_service_instance(mock_llm_config_manager, sample_ollama_provider_config, sample_llama2_model_config):
    """Provides an OllamaService instance with mocked config for a specific model."""
    model_id_key = "llama2_test" # Key used in config, not the Ollama model tag

    mock_llm_config_manager.get_provider_config.return_value = sample_ollama_provider_config
    mock_llm_config_manager.get_model_config.return_value = sample_llama2_model_config

    return OllamaService(model_id=model_id_key, config_manager=mock_llm_config_manager)

@pytest.fixture
def sample_llm_context():
    """Provides a sample LLMContext instance."""
    context = LLMContext(system_prompt="You are a helpful test bot.")
    context.add_user_message("First user message")
    context.add_assistant_message("First assistant response")
    return context

# --- Test Cases ---

class TestOllamaServiceInitialization:
    def test_successful_initialization(self, ollama_service_instance, sample_ollama_provider_config, sample_llama2_model_config):
        assert ollama_service_instance.ollama_model_name == sample_llama2_model_config.model_name
        assert ollama_service_instance.base_url == sample_ollama_provider_config.base_url
        assert ollama_service_instance.api_endpoint == f"{sample_ollama_provider_config.base_url}/api/chat"
        assert isinstance(ollama_service_instance.client, httpx.AsyncClient)
        # Check merged default settings
        assert ollama_service_instance.default_settings.temperature == 0.6 # From model
        assert ollama_service_instance.default_settings.num_ctx == 2048    # From model
        assert ollama_service_instance.default_settings.keep_alive == "5m" # Model overrides provider

    def test_initialization_default_base_url(self, mock_llm_config_manager, sample_llama2_model_config):
        provider_config_no_base_url = LLMProviderConfig(provider_name="ollama", models={})
        mock_llm_config_manager.get_provider_config.return_value = provider_config_no_base_url
        mock_llm_config_manager.get_model_config.return_value = sample_llama2_model_config

        service = OllamaService(model_id="llama2_test", config_manager=mock_llm_config_manager)
        assert service.base_url == DEFAULT_OLLAMA_BASE_URL
        assert service.api_endpoint == f"{DEFAULT_OLLAMA_BASE_URL}/api/chat"

    def test_initialization_provider_config_not_found(self, mock_llm_config_manager):
        mock_llm_config_manager.get_provider_config.return_value = None
        with pytest.raises(LLMConfigError, match="Ollama provider configuration not found."):
            OllamaService(model_id="any_model", config_manager=mock_llm_config_manager)

    def test_initialization_model_config_not_found(self, mock_llm_config_manager, sample_ollama_provider_config):
        mock_llm_config_manager.get_provider_config.return_value = sample_ollama_provider_config
        mock_llm_config_manager.get_model_config.return_value = None
        model_id_key = "unknown_model"
        with pytest.raises(LLMConfigError, match=f"Configuration for Ollama model ID '{model_id_key}' not found."):
            OllamaService(model_id=model_id_key, config_manager=mock_llm_config_manager)

    def test_initialization_merges_keep_alive_correctly(self, mock_llm_config_manager):
        # Provider has default, model does not
        provider_conf = LLMProviderConfig(provider_name="ollama", default_keep_alive="15m", models={})
        model_conf = LLMModelConfig(model_name="model1", default_settings=LLMModelSettings(temperature=0.5))
        mock_llm_config_manager.get_provider_config.return_value = provider_conf
        mock_llm_config_manager.get_model_config.return_value = model_conf
        service1 = OllamaService(model_id="m1", config_manager=mock_llm_config_manager)
        assert service1.default_settings.keep_alive == "15m"

        # Model has default, provider does not
        provider_conf_no_ka = LLMProviderConfig(provider_name="ollama", models={})
        model_conf_with_ka = LLMModelConfig(model_name="model2", default_settings=LLMModelSettings(keep_alive="3m"))
        mock_llm_config_manager.get_provider_config.return_value = provider_conf_no_ka
        mock_llm_config_manager.get_model_config.return_value = model_conf_with_ka
        service2 = OllamaService(model_id="m2", config_manager=mock_llm_config_manager)
        assert service2.default_settings.keep_alive == "3m"

        # Neither has default, should use OLLAMA_DEFAULT_KEEP_ALIVE from ollama_service.py
        # The logic in OllamaService actually defaults to LLMModelSettings default if None from provider/model.
        # The DEFAULT_KEEP_ALIVE in OllamaService is used if LLMModelSettings.keep_alive is also None.
        # LLMModelSettings.keep_alive is None by default.
        # OllamaService's internal default_settings merging logic seems to prioritize model, then provider.
        # If both are None, then the global DEFAULT_KEEP_ALIVE from ollama_service.py is used.
        model_conf_no_ka = LLMModelConfig(model_name="model3")
        mock_llm_config_manager.get_provider_config.return_value = provider_conf_no_ka
        mock_llm_config_manager.get_model_config.return_value = model_conf_no_ka
        service3 = OllamaService(model_id="m3", config_manager=mock_llm_config_manager)
        # This depends on the exact implementation details of default merging in OllamaService.
        # The code is: final_options.pop("keep_alive", self.default_settings.keep_alive or DEFAULT_KEEP_ALIVE)
        # self.default_settings.keep_alive is what we are testing here.
        # self.default_settings is LLMModelSettings(**merged_defaults)
        # merged_defaults comes from provider_default_settings then model_default_settings.
        # provider_default_settings = LLMModelSettings(keep_alive=provider_config.default_keep_alive or DEFAULT_KEEP_ALIVE)
        # In this case, provider_config.default_keep_alive is None, so it uses DEFAULT_KEEP_ALIVE ("5m") from ollama_service.py
        from llm_service.ollama_service import DEFAULT_KEEP_ALIVE as OLLAMA_SERVICE_DEFAULT_KA
        assert service3.default_settings.keep_alive == OLLAMA_SERVICE_DEFAULT_KA


class TestOllamaServiceMethods:
    def test_get_service_name(self, ollama_service_instance):
        # model_id_key used in fixture is "llama2_test"
        assert ollama_service_instance.get_service_name() == "ollama_llama2_test"

    @pytest.mark.asyncio
    async def test_close_method(self, ollama_service_instance):
        with patch.object(ollama_service_instance.client, 'aclose', new_callable=AsyncMock) as mock_aclose:
            await ollama_service_instance.close()
            mock_aclose.assert_called_once()

@pytest.mark.asyncio
class TestOllamaServiceGenerateResponse:

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_success(self, mock_post, ollama_service_instance, sample_llm_context, sample_llama2_model_config):
        mock_response_content = "Test response from Ollama."
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_api_response = httpx.Response(
            200,
            json={"model": sample_llama2_model_config.model_name, "created_at": "...",
                  "message": {"role": "assistant", "content": mock_response_content}, "done": True},
            request=dummy_request
        )
        mock_post.return_value = mock_api_response

        prompt = "User prompt"
        runtime_settings = {"temperature": 0.8, "num_predict": 100} # num_predict is an Ollama option

        response = await ollama_service_instance.generate_response(prompt, context=sample_llm_context, settings=runtime_settings)
        assert response == mock_response_content

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == ollama_service_instance.api_endpoint

        payload = call_args.kwargs['json']
        assert payload["model"] == sample_llama2_model_config.model_name
        expected_messages = [
            {"role": "system", "content": sample_llm_context.initial_system_prompt},
            {"role": "user", "content": "First user message"},
            {"role": "assistant", "content": "First assistant response"},
            {"role": "user", "content": prompt}
        ]
        assert payload["messages"] == expected_messages
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == runtime_settings["temperature"]
        assert payload["options"]["num_ctx"] == sample_llama2_model_config.default_settings.num_ctx # Default
        assert payload["options"]["num_predict"] == runtime_settings["num_predict"] # Runtime
        assert payload["keep_alive"] == sample_llama2_model_config.default_settings.keep_alive # From model default

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_empty_context(self, mock_post, ollama_service_instance, sample_llama2_model_config):
        mock_response_content = "Response with no context."
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_api_response = httpx.Response(
            200,
            json={"model": "test", "message": {"role": "assistant", "content": mock_response_content}, "done": True},
            request=dummy_request
        )
        mock_post.return_value = mock_api_response

        prompt = "User prompt, no context"
        empty_context = LLMContext() # No system prompt
        response = await ollama_service_instance.generate_response(prompt, context=empty_context)

        assert response == mock_response_content
        payload = mock_post.call_args.kwargs['json']
        assert payload["messages"] == [{"role": "user", "content": prompt}]
        assert payload["options"]["temperature"] == sample_llama2_model_config.default_settings.temperature

    @pytest.mark.parametrize(
        "status_code, error_json, expected_detail_part",
        [
            (400, {"error": "bad request data"}, "bad request data"),
            (404, {"error": "model 'mistake' not found"}, "model 'mistake' not found"),
            (500, {"error": "internal server error"}, "internal server error"),
        ]
    )
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_api_errors(self, mock_post, status_code, error_json, expected_detail_part, ollama_service_instance):
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(status_code, json=error_json, request=dummy_request)

        # OllamaService extracts the 'error' string from the JSON, not the full JSON string in the message.
        expected_match_regex = f"Ollama API request failed with status {status_code}: {expected_detail_part}.*Provider: {ollama_service_instance.get_service_name()}.*Status Code: {status_code}"
        with pytest.raises(LLMResponseError, match=expected_match_regex):
            await ollama_service_instance.generate_response("prompt")

    @pytest.mark.parametrize(
        "exception_type, expected_message_part",
        [
            (httpx.TimeoutException, "Request to Ollama API timed out: Mocked network error"),
            (httpx.ConnectError, "An error occurred during the request to Ollama API: Mocked network error"),
            (httpx.RequestError, "An error occurred during the request to Ollama API: Mocked network error"),
        ]
    )
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_network_errors(self, mock_post, exception_type, expected_message_part, ollama_service_instance):
        mock_post.side_effect = exception_type("Mocked network error")

        with pytest.raises(LLMResponseError, match=expected_message_part):
            await ollama_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_malformed_json_success(self, mock_post, ollama_service_instance):
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, text="not valid json", request=dummy_request)
        with pytest.raises(LLMResponseError, match="Failed to decode JSON response from Ollama API"):
            await ollama_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_missing_message_in_json(self, mock_post, ollama_service_instance):
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, json={"model": "test", "done": True}, request=dummy_request)
        with pytest.raises(LLMResponseError, match="Invalid response structure from Ollama API: 'message.content' missing."):
            await ollama_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_missing_content_in_message(self, mock_post, ollama_service_instance):
        dummy_request = httpx.Request("POST", ollama_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, json={"model": "test", "message": {"role": "assistant"}}, request=dummy_request)
        with pytest.raises(LLMResponseError, match="Invalid response structure from Ollama API: 'message.content' missing."):
            await ollama_service_instance.generate_response("prompt")
