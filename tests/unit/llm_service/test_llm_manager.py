import pytest
import logging
from unittest.mock import patch, MagicMock, AsyncMock, call

from llm_service.manager import LLMManager
from llm_service.config import LLMConfigManager, LLMProviderConfig, LLMModelConfig, LLMModelSettings, LLMProvidersRoot
from llm_service.base import AbstractLLMService, LLMContext
from llm_service.exceptions import LLMModelNotFoundError, LLMConfigError
from llm_service.openai_service import OpenAIService # Needed for isinstance checks and patching
from llm_service.ollama_service import OllamaService # Needed for isinstance checks and patching

# --- Fixtures ---

@pytest.fixture
def mock_llm_config_manager():
    """Provides a MagicMock for LLMConfigManager."""
    manager = MagicMock(spec=LLMConfigManager)
    manager.get_default_model_id.return_value = None # Default, can be overridden in tests
    manager.get_all_providers_config.return_value = {} # Default, can be overridden
    return manager

# --- Helper to create mock service instances ---
def create_mock_service(name="mock_service"):
    service = MagicMock(spec=AbstractLLMService)
    service.get_service_name.return_value = name
    service.generate_response = AsyncMock(return_value="mocked response from " + name)
    service.close = AsyncMock()
    return service

# --- Test Cases ---

class TestLLMManagerInitialization:

    @patch('llm_service.manager.OpenAIService', autospec=True)
    @patch('llm_service.manager.OllamaService', autospec=True)
    def test_successful_loading_and_default_model(self, MockOllamaService, MockOpenAIService, mock_llm_config_manager):
        # Mock service instances that constructors would return
        mock_openai_instance = create_mock_service("openai_gpt_test")
        MockOpenAIService.return_value = mock_openai_instance

        mock_ollama_instance = create_mock_service("ollama_llama_test")
        MockOllamaService.return_value = mock_ollama_instance

        # Setup mock config manager
        openai_model_conf = LLMModelConfig(model_name="gpt-actual-name")
        ollama_model_conf = LLMModelConfig(model_name="llama-actual-name")

        providers_config_data = {
            "openai": LLMProviderConfig(models={"gpt_test": openai_model_conf}),
            "ollama": LLMProviderConfig(models={"llama_test": ollama_model_conf})
        }
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data
        mock_llm_config_manager.get_default_model_id.return_value = "openai_gpt_test"

        llm_manager = LLMManager(config_manager=mock_llm_config_manager)

        MockOpenAIService.assert_called_once_with(model_id="gpt_test", config_manager=mock_llm_config_manager)
        MockOllamaService.assert_called_once_with(model_id="llama_test", config_manager=mock_llm_config_manager)

        assert len(llm_manager.services) == 2
        assert llm_manager.services["openai_gpt_test"] == mock_openai_instance
        assert llm_manager.services["ollama_llama_test"] == mock_ollama_instance
        assert llm_manager.get_active_model_id() == "openai_gpt_test"

    @patch('llm_service.manager.OpenAIService', autospec=True)
    @patch('llm_service.manager.OllamaService', autospec=True)
    def test_loading_with_no_default_model_id(self, MockOllamaService, MockOpenAIService, mock_llm_config_manager):
        mock_openai_instance = create_mock_service("openai_gpt_test")
        MockOpenAIService.return_value = mock_openai_instance

        providers_config_data = {
            "openai": LLMProviderConfig(models={"gpt_test": LLMModelConfig(model_name="gpt-actual")})
        }
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data
        mock_llm_config_manager.get_default_model_id.return_value = None # No default

        llm_manager = LLMManager(config_manager=mock_llm_config_manager)

        # Should pick the first available if no default
        assert llm_manager.get_active_model_id() == "openai_gpt_test"

    @patch('llm_service.manager.OpenAIService', autospec=True)
    @patch('llm_service.manager.OllamaService', autospec=True)
    def test_loading_with_invalid_default_model_id(self, MockOllamaService, MockOpenAIService, mock_llm_config_manager, caplog):
        mock_openai_instance = create_mock_service("openai_gpt_test")
        MockOpenAIService.return_value = mock_openai_instance

        providers_config_data = {
            "openai": LLMProviderConfig(models={"gpt_test": LLMModelConfig(model_name="gpt-actual")})
        }
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data
        mock_llm_config_manager.get_default_model_id.return_value = "non_existent_default"

        with caplog.at_level(logging.WARNING):
            llm_manager = LLMManager(config_manager=mock_llm_config_manager)

        assert "Default model ID 'non_existent_default' not found" in caplog.text
        # Active model should be None as the default was invalid and no fallback to first is implemented in this specific path
        assert llm_manager.get_active_model_id() is None


    @patch('llm_service.manager.OpenAIService', autospec=True)
    @patch('llm_service.manager.logger', autospec=True) # Patch module-level logger instance
    def test_service_instantiation_error(self, mock_logger_in_manager, MockOpenAIService, mock_llm_config_manager):
        MockOpenAIService.side_effect = LLMConfigError("Failed to init OpenAI")

        providers_config_data = {
            "openai": LLMProviderConfig(models={"gpt_test": LLMModelConfig(model_name="gpt-actual")})
        }
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data

        llm_manager = LLMManager(config_manager=mock_llm_config_manager)

        assert "openai_gpt_test" not in llm_manager.services
        assert len(llm_manager.services) == 0
        mock_logger_in_manager.error.assert_any_call("Configuration error loading model 'openai_gpt_test': Failed to init OpenAI. This model will be unavailable.")


    @patch('llm_service.manager.logger', autospec=True) # Patch module-level logger instance
    def test_unsupported_provider_type(self, mock_logger_in_manager, mock_llm_config_manager):
        providers_config_data = {
            "new_fancy_provider": LLMProviderConfig(models={"fancy_model": LLMModelConfig(model_name="fancy-actual")})
        }
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data

        llm_manager = LLMManager(config_manager=mock_llm_config_manager)

        assert "new_fancy_provider_fancy_model" not in llm_manager.services
        mock_logger_in_manager.warning.assert_any_call("Unsupported LLM provider type: new_fancy_provider for model key fancy_model. Skipping.")

    def test_no_providers_configured(self, mock_llm_config_manager, caplog):
        mock_llm_config_manager.get_all_providers_config.return_value = {}
        with caplog.at_level(logging.WARNING):
            llm_manager = LLMManager(config_manager=mock_llm_config_manager)
        assert "No providers found in the LLM configuration." in caplog.text
        assert llm_manager.services == {}
        assert llm_manager.get_active_model_id() is None

    def test_provider_with_no_models(self, mock_llm_config_manager, caplog):
        providers_config_data = {"openai": LLMProviderConfig(models={})}
        mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data
        with caplog.at_level(logging.WARNING):
            llm_manager = LLMManager(config_manager=mock_llm_config_manager)
        assert "Provider 'openai' has no models configured. Skipping." in caplog.text
        assert llm_manager.services == {}


class TestLLMManagerServiceAccess:

    @pytest.fixture
    def populated_manager(self, mock_llm_config_manager):
        # Helper to create a manager with some mock services
        self.mock_service1 = create_mock_service("service1")
        self.mock_service2 = create_mock_service("service2")

        # Patch constructors to return these mocks
        with patch('llm_service.manager.OpenAIService', return_value=self.mock_service1), \
             patch('llm_service.manager.OllamaService', return_value=self.mock_service2):

            providers_config_data = {
                "openai": LLMProviderConfig(models={"s1": LLMModelConfig(model_name="s1_actual")}),
                "ollama": LLMProviderConfig(models={"s2": LLMModelConfig(model_name="s2_actual")})
            }
            mock_llm_config_manager.get_all_providers_config.return_value = providers_config_data
            mock_llm_config_manager.get_default_model_id.return_value = "openai_s1"

            manager = LLMManager(config_manager=mock_llm_config_manager)
            # Ensure services dict is correctly keyed for tests
            manager.services = {"openai_s1": self.mock_service1, "ollama_s2": self.mock_service2}
            manager.active_model_id = "openai_s1"
            return manager

    def test_get_service_active(self, populated_manager):
        assert populated_manager.get_service() == self.mock_service1

    def test_get_service_by_id(self, populated_manager):
        assert populated_manager.get_service("ollama_s2") == self.mock_service2

    def test_get_service_non_existent(self, populated_manager):
        assert populated_manager.get_service("non_existent_service") is None

    def test_set_active_model_success(self, populated_manager):
        populated_manager.set_active_model("ollama_s2")
        assert populated_manager.get_active_model_id() == "ollama_s2"
        assert populated_manager.get_service() == self.mock_service2

    def test_set_active_model_not_found(self, populated_manager):
        with pytest.raises(LLMModelNotFoundError, match="Cannot set active model. Model ID 'non_existent' not found among loaded services."):
            populated_manager.set_active_model("non_existent")

    def test_list_available_models(self, populated_manager):
        available_models = populated_manager.list_available_models()
        assert sorted(available_models) == sorted(["openai_s1", "ollama_s2"])

    @pytest.mark.asyncio
    async def test_close_all_services(self, populated_manager):
        await populated_manager.close_all_services()
        self.mock_service1.close.assert_called_once()
        self.mock_service2.close.assert_called_once()


@pytest.mark.asyncio
class TestLLMManagerGenerateResponse:

    @pytest.fixture
    def manager_with_mocked_active_service(self, mock_llm_config_manager):
        manager = LLMManager(config_manager=mock_llm_config_manager) # Will be empty initially
        self.active_service_mock = create_mock_service("active_mock_service")
        manager.services = {"active_model": self.active_service_mock}
        manager.active_model_id = "active_model"
        return manager

    async def test_generate_response_uses_active_service(self, manager_with_mocked_active_service):
        prompt = "test prompt"
        context = LLMContext()
        settings = {"temp": 0.5}

        response = await manager_with_mocked_active_service.generate_response(prompt, context, settings)

        assert response == "mocked response from active_mock_service"
        self.active_service_mock.generate_response.assert_awaited_once_with(prompt, context, settings)

    async def test_generate_response_uses_specified_service(self, mock_llm_config_manager):
        # Create manager and add two services
        manager = LLMManager(config_manager=mock_llm_config_manager) # Empty
        service1_mock = create_mock_service("service1")
        service2_mock = create_mock_service("service2")
        manager.services = {"s1": service1_mock, "s2": service2_mock}
        manager.active_model_id = "s1" # s1 is active

        prompt = "another test"
        await manager.generate_response(prompt, model_id="s2") # Request s2

        service1_mock.generate_response.assert_not_called()
        service2_mock.generate_response.assert_awaited_once_with(prompt, None, None)


    async def test_generate_response_model_not_found(self, manager_with_mocked_active_service):
        with pytest.raises(LLMModelNotFoundError, match="Cannot generate response. LLM service for model ID 'non_existent_model' is not available."):
            await manager_with_mocked_active_service.generate_response("prompt", model_id="non_existent_model")

    async def test_generate_response_no_active_model_and_none_specified(self, mock_llm_config_manager):
        manager = LLMManager(config_manager=mock_llm_config_manager) # Empty services, no active model
        manager.active_model_id = None

        # The message part will say 'None' because target_model_id becomes None, but the model_id attribute of exception is 'N/A'
        with pytest.raises(LLMModelNotFoundError, match="Cannot generate response. LLM service for model ID 'None' is not available."):
            await manager.generate_response("prompt")
