import pytest
import httpx
import os
from unittest.mock import patch, AsyncMock, MagicMock

from llm_service.openai_service import OpenAIService
from llm_service.base import LLMContext
from llm_service.config import LLMConfigManager, LLMProviderConfig, LLMModelConfig, LLMModelSettings
from llm_service.exceptions import LLMConfigError, LLMResponseError

# --- Fixtures ---

@pytest.fixture
def mock_llm_config_manager():
    """Provides a MagicMock for LLMConfigManager."""
    mock_manager = MagicMock(spec=LLMConfigManager)
    return mock_manager

@pytest.fixture
def sample_openai_provider_config():
    return LLMProviderConfig(
        provider_name="openai",
        api_key="test_openai_api_key",
        models={} # Models will be added by specific tests or fixtures
    )

@pytest.fixture
def sample_gpt_3_5_turbo_model_config():
    return LLMModelConfig(
        model_name="gpt-3.5-turbo", # This is the actual OpenAI model name
        default_settings=LLMModelSettings(temperature=0.7, max_tokens=150)
    )

@pytest.fixture
def openai_service_instance(mock_llm_config_manager, sample_openai_provider_config, sample_gpt_3_5_turbo_model_config):
    """Provides an OpenAIService instance with mocked config for a specific model."""
    model_id_key = "gpt_3_5_turbo_test" # This is the key used in config, not the OpenAI model name

    # Configure the mock_llm_config_manager to return the sample configs
    mock_llm_config_manager.get_provider_config.return_value = sample_openai_provider_config
    mock_llm_config_manager.get_model_config.return_value = sample_gpt_3_5_turbo_model_config

    return OpenAIService(model_id=model_id_key, config_manager=mock_llm_config_manager)

@pytest.fixture
def sample_llm_context():
    """Provides a sample LLMContext instance."""
    context = LLMContext(system_prompt="You are a test assistant.")
    context.add_user_message("Hello there!")
    context.add_assistant_message("General Kenobi!")
    return context

# --- Test Cases ---

class TestOpenAIServiceInitialization:
    def test_successful_initialization(self, openai_service_instance, sample_gpt_3_5_turbo_model_config):
        assert openai_service_instance.model_name == sample_gpt_3_5_turbo_model_config.model_name
        assert openai_service_instance.api_key == "test_openai_api_key"
        assert openai_service_instance.api_endpoint == "https://api.openai.com/v1/chat/completions" # Corrected attribute name
        assert isinstance(openai_service_instance.client, httpx.AsyncClient) # Corrected attribute name

    def test_initialization_missing_api_key(self, mock_llm_config_manager, sample_gpt_3_5_turbo_model_config):
        model_id_key = "test_model_missing_key"

        # Provider config that results in a None API key
        provider_config_no_key = LLMProviderConfig(provider_name="openai", api_key=None, models={})

        mock_llm_config_manager.get_provider_config.return_value = provider_config_no_key
        mock_llm_config_manager.get_model_config.return_value = sample_gpt_3_5_turbo_model_config

        expected_error_msg = "OpenAI API key not found. Please set it directly or via environment variable 'OPENAI_API_KEY'."
        with pytest.raises(LLMConfigError, match=expected_error_msg):
            OpenAIService(model_id=model_id_key, config_manager=mock_llm_config_manager)

    def test_initialization_provider_config_not_found(self, mock_llm_config_manager):
        mock_llm_config_manager.get_provider_config.return_value = None
        with pytest.raises(LLMConfigError, match="OpenAI provider configuration not found."):
            OpenAIService(model_id="any_model", config_manager=mock_llm_config_manager)

    def test_initialization_model_config_not_found(self, mock_llm_config_manager, sample_openai_provider_config):
        mock_llm_config_manager.get_provider_config.return_value = sample_openai_provider_config
        mock_llm_config_manager.get_model_config.return_value = None
        model_id_key = "unknown_model"
        with pytest.raises(LLMConfigError, match=f"Configuration for OpenAI model ID '{model_id_key}' not found."):
            OpenAIService(model_id=model_id_key, config_manager=mock_llm_config_manager)


class TestOpenAIServiceMethods:

    def test_get_service_name(self, openai_service_instance, sample_gpt_3_5_turbo_model_config):
        # The service name is f"openai_{model_id_key}" passed to init
        # model_id_key used in openai_service_instance fixture is "gpt_3_5_turbo_test"
        assert openai_service_instance.get_service_name() == f"openai_gpt_3_5_turbo_test"

    @pytest.mark.asyncio
    async def test_close_method(self, openai_service_instance):
        with patch.object(openai_service_instance.client, 'aclose', new_callable=AsyncMock) as mock_aclose: # Corrected attribute name
            await openai_service_instance.close()
            mock_aclose.assert_called_once()

@pytest.mark.asyncio
class TestOpenAIServiceGenerateResponse:

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_success(self, mock_post, openai_service_instance, sample_llm_context, sample_gpt_3_5_turbo_model_config):
        mock_response_content = "This is a test response from OpenAI."
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_api_response = httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": mock_response_content}}]},
            request=dummy_request
        )
        mock_post.return_value = mock_api_response

        prompt = "Test prompt"
        runtime_settings = {"temperature": 0.5, "top_p": 0.9}

        response = await openai_service_instance.generate_response(prompt, context=sample_llm_context, settings=runtime_settings)

        assert response == mock_response_content

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == openai_service_instance.api_endpoint # Corrected attribute

        headers = call_args.kwargs['headers']
        assert headers["Authorization"] == f"Bearer {openai_service_instance.api_key}"

        payload = call_args.kwargs['json']
        assert payload["model"] == sample_gpt_3_5_turbo_model_config.model_name

        expected_messages = [
            {"role": "system", "content": sample_llm_context.initial_system_prompt},
            {"role": "user", "content": "Hello there!"},
            {"role": "assistant", "content": "General Kenobi!"},
            {"role": "user", "content": prompt}
        ]
        assert payload["messages"] == expected_messages
        assert payload["temperature"] == runtime_settings["temperature"] # Runtime overrides default
        assert payload["max_tokens"] == sample_gpt_3_5_turbo_model_config.default_settings.max_tokens # Default used
        assert payload["top_p"] == runtime_settings["top_p"] # Additional runtime setting

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_empty_context_and_settings(self, mock_post, openai_service_instance, sample_gpt_3_5_turbo_model_config):
        mock_response_content = "Minimal response."
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_api_response = httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": mock_response_content}}]},
            request=dummy_request
        )
        mock_post.return_value = mock_api_response

        prompt = "Minimal prompt"
        empty_context = LLMContext() # No system prompt, no history

        response = await openai_service_instance.generate_response(prompt, context=empty_context, settings=None)

        assert response == mock_response_content

        payload = mock_post.call_args.kwargs['json']
        assert payload["messages"] == [{"role": "user", "content": prompt}]
        assert payload["temperature"] == sample_gpt_3_5_turbo_model_config.default_settings.temperature # Default
        assert payload["max_tokens"] == sample_gpt_3_5_turbo_model_config.default_settings.max_tokens   # Default

    @pytest.mark.parametrize(
        "status_code, error_payload, expected_message_detail",
        [
            (400, {"error": {"message": "Bad request", "type": "invalid_request_error"}}, "Bad request"),
            (401, {"error": {"message": "Invalid API key", "type": "authentication_error"}}, "Invalid API key"),
            (403, {"error": {"message": "Permission denied", "type": "permission_error"}}, "Permission denied"),
            (429, {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}, "Rate limit exceeded"),
            (500, {"error": {"message": "Internal server error", "type": "server_error"}}, "Internal server error"),
            (503, {"error": {"message": "Service unavailable", "type": "service_unavailable_error"}}, "Service unavailable"),
        ]
    )
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_api_errors(self, mock_post, status_code, error_payload, expected_message_detail, openai_service_instance):
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(status_code, json=error_payload, request=dummy_request)

        # Regex needs to be more flexible to include the full error detail string (which is a dict here)
        # and the provider name / status code suffix that LLMResponseError adds.
        expected_match_regex = f"OpenAI API request failed with status {status_code}: {error_payload}.*Provider: {openai_service_instance.get_service_name()}.*Status Code: {status_code}"
        with pytest.raises(LLMResponseError, match=expected_match_regex):
            await openai_service_instance.generate_response("prompt")

    @pytest.mark.parametrize(
        "exception_type, expected_message_part",
        [
            (httpx.TimeoutException, "Request to OpenAI API timed out: Mocked network error"),
            (httpx.ConnectError, "An error occurred during the request to OpenAI API: Mocked network error"), # Adjusted to match generic RequestError
            (httpx.RequestError, "An error occurred during the request to OpenAI API: Mocked network error"), # Generic
        ]
    )
    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_network_errors(self, mock_post, exception_type, expected_message_part, openai_service_instance):
        mock_post.side_effect = exception_type("Mocked network error")

        with pytest.raises(LLMResponseError, match=expected_message_part):
            await openai_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_malformed_json_success(self, mock_post, openai_service_instance):
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, text="not a valid json", request=dummy_request)
        with pytest.raises(LLMResponseError, match="Failed to decode JSON response from OpenAI API"):
            await openai_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_missing_choices_in_json(self, mock_post, openai_service_instance):
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, json={"some_other_key": "value"}, request=dummy_request) # choices is missing
        expected_match_regex = "Invalid response structure from OpenAI API.*Provider: " + openai_service_instance.get_service_name()
        with pytest.raises(LLMResponseError, match=expected_match_regex):
            await openai_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_missing_message_in_choice(self, mock_post, openai_service_instance):
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, json={"choices": [{"no_message_here": "..."}]}, request=dummy_request)
        expected_match_regex = "Invalid response structure from OpenAI API.*Provider: " + openai_service_instance.get_service_name()
        with pytest.raises(LLMResponseError, match=expected_match_regex):
            await openai_service_instance.generate_response("prompt")

    @patch('httpx.AsyncClient.post', new_callable=AsyncMock)
    async def test_generate_response_missing_content_in_message(self, mock_post, openai_service_instance):
        dummy_request = httpx.Request("POST", openai_service_instance.api_endpoint)
        mock_post.return_value = httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "no_content": "..."}}]}, request=dummy_request)
        expected_match_regex = "Invalid response structure from OpenAI API.*Provider: " + openai_service_instance.get_service_name()
        with pytest.raises(LLMResponseError, match=expected_match_regex):
            await openai_service_instance.generate_response("prompt")
