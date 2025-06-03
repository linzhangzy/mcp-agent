import logging
from typing import Dict, Optional, List, Any

from .base import AbstractLLMService
from .config import LLMConfigManager
from .exceptions import LLMModelNotFoundError, LLMConfigError
from .openai_service import OpenAIService
from .ollama_service import OllamaService
from .base import LLMContext # Moved import to the top

logger = logging.getLogger(__name__)

class LLMManager:
    """
    Manages the instantiation, tracking, and switching of LLM services.
    """
    def __init__(self, config_manager: LLMConfigManager):
        """
        Initializes the LLMManager.

        Args:
            config_manager: An instance of LLMConfigManager that has loaded
                            LLM configurations.
        """
        self.config_manager: LLMConfigManager = config_manager
        self.services: Dict[str, AbstractLLMService] = {}
        self.active_model_id: Optional[str] = None

        self._load_configured_services()

        default_model_id_from_config = self.config_manager.get_default_model_id()
        if default_model_id_from_config:
            if default_model_id_from_config in self.services:
                self.active_model_id = default_model_id_from_config
                logger.info(f"Default active model set to: {self.active_model_id}")
            else:
                logger.warning(
                    f"Default model ID '{default_model_id_from_config}' not found among loaded services. "
                    f"Available: {list(self.services.keys())}. No active model set initially."
                )
        elif self.services:
            # If no default is specified, but services were loaded, pick the first one as a fallback.
            self.active_model_id = next(iter(self.services))
            logger.info(f"No default model specified. Active model set to first available: {self.active_model_id}")
        else:
            logger.warning("No LLM services loaded and no default model ID specified.")

    def _load_configured_services(self) -> None:
        """
        Loads and instantiates LLM services based on the configurations.
        Populates the self.services dictionary.
        The key for self.services will be f"{provider_name}_{model_key_in_config}".
        """
        logger.info("Loading configured LLM services...")
        all_providers_config = self.config_manager.get_all_providers_config()

        if not all_providers_config:
            logger.warning("No providers found in the LLM configuration.")
            return

        for provider_name, provider_config in all_providers_config.items():
            if not provider_config.models:
                logger.warning(f"Provider '{provider_name}' has no models configured. Skipping.")
                continue

            for model_key, model_config in provider_config.models.items():
                # model_key is the key from the config, e.g., "gpt_4o_mini", "test_model" for ollama
                # model_config.model_name is the actual name passed to the service, e.g. "gpt-4o-mini", "llama3:8b"

                service_key = f"{provider_name}_{model_key}" # This becomes the unique ID in self.services

                try:
                    service_instance: Optional[AbstractLLMService] = None
                    if provider_name.lower() == "openai":
                        service_instance = OpenAIService(model_id=model_key, config_manager=self.config_manager)
                    elif provider_name.lower() == "ollama":
                        service_instance = OllamaService(model_id=model_key, config_manager=self.config_manager)
                    else:
                        logger.warning(f"Unsupported LLM provider type: {provider_name} for model key {model_key}. Skipping.")
                        continue

                    if service_instance:
                        # Use the service_key derived above for consistency with default_model_id format
                        self.services[service_key] = service_instance
                        logger.info(f"Successfully loaded service: {service_key} (Service Name: {service_instance.get_service_name()})")

                except LLMConfigError as e:
                    logger.error(f"Configuration error loading model '{service_key}': {e}. This model will be unavailable.")
                except Exception as e:
                    logger.error(f"Failed to load LLM service for model key '{model_key}' from provider '{provider_name}': {e}", exc_info=True)

        logger.info(f"LLM services loading complete. Loaded: {list(self.services.keys())}")


    def get_service(self, model_id: Optional[str] = None) -> Optional[AbstractLLMService]:
        """
        Retrieves an LLM service instance.

        Args:
            model_id: The unique ID of the model service to retrieve (e.g., "openai_gpt_4o_mini").
                      If None, the currently active service is returned.

        Returns:
            The AbstractLLMService instance, or None if not found.
        """
        target_model_id = model_id if model_id is not None else self.active_model_id
        if not target_model_id:
            logger.warning("get_service called with no model_id and no active model set.")
            return None

        service = self.services.get(target_model_id)
        if not service:
            logger.warning(f"Service for model ID '{target_model_id}' not found.")
        return service

    def set_active_model(self, model_id: str) -> None:
        """
        Sets the active LLM service.

        Args:
            model_id: The unique ID of the model to set as active.
                      Must be one of the keys in self.services.

        Raises:
            LLMModelNotFoundError: If the model_id is not found in loaded services.
        """
        if model_id not in self.services:
            logger.error(f"Attempted to set active model to '{model_id}', which is not loaded. Available: {list(self.services.keys())}")
            raise LLMModelNotFoundError(model_id=model_id, message=f"Cannot set active model. Model ID '{model_id}' not found among loaded services.")

        self.active_model_id = model_id
        logger.info(f"Active LLM model set to: {model_id}")

    def get_active_model_id(self) -> Optional[str]:
        """Returns the ID of the currently active LLM model."""
        return self.active_model_id

    def list_available_models(self) -> List[str]:
        """Returns a list of unique IDs of all successfully loaded models."""
        return list(self.services.keys())

    async def close_all_services(self) -> None:
        """
        Gracefully closes all instantiated LLM services.
        This should be called on application shutdown to release resources,
        especially network clients.
        """
        logger.info("Closing all LLM services...")
        for model_id, service in self.services.items():
            try:
                if hasattr(service, 'close') and callable(service.close):
                    await service.close()
                    logger.info(f"Closed service: {model_id}")
            except Exception as e:
                logger.error(f"Error closing service {model_id}: {e}", exc_info=True)
        logger.info("All LLM services closed.")

    async def generate_response(
        self,
        prompt: str,
        context: Optional[LLMContext] = None, # LLMContext needs to be imported or defined
        settings: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None
    ) -> str:
        """
        Generates a response using either a specified LLM model or the active one.

        Args:
            prompt: The user's prompt.
            context: Optional conversation context.
            settings: Optional provider-specific settings.
            model_id: Optional model ID to use. If None, uses the active model.

        Returns:
            The LLM's response as a string.

        Raises:
            LLMModelNotFoundError: If the specified or active model is not available.
            LLMResponseError: If the underlying service encounters an error.
        """
        service = self.get_service(model_id)
        if not service:
            target_model_id = model_id if model_id is not None else self.active_model_id
            raise LLMModelNotFoundError(model_id=target_model_id or "N/A", message=f"Cannot generate response. LLM service for model ID '{target_model_id}' is not available.")

        # LLMContext is used by services, ensure it's available in this scope
        # from .base import LLMContext # Or ensure it's imported at the top
        return await service.generate_response(prompt, context, settings)

# Example usage (conceptual, requires async context to run)
async def _example_manager_usage():
    pass # Added pass to provide a body for the function
    # Assume llm_config.yaml exists and is populated
    # from .config import LLMConfigManager # already imported
    # from pathlib import Path # For path manipulation
    # config_path = Path(__file__).parent / "llm_config.yaml"
    # if not config_path.exists():
    #     print(f"Error: Config file not found at {config_path}")
    #     # Create a dummy one for testing if needed
    #     dummy_config_content = """
    # default_model_id: "openai_default"
    # global_request_timeout: 60
    # llm_providers:
    #   openai:
    #     api_key_env_var: "OPENAI_API_KEY" # Ensure this env var is set
    #     models:
    #       default: # model_id used in default_model_id
    #         model_name: "gpt-3.5-turbo"
    #       gpt4:
    #         model_name: "gpt-4"
    #   ollama:
    #     base_url: "http://localhost:11434"
    #     models:
    #       llama2_chat: # model_id
    #         model_name: "llama2:chat" # Ollama model tag
    # """
    #     # with open(config_path, "w") as f:
    #     #     f.write(dummy_config_content)
    #     # print(f"Created dummy config at {config_path}, please set OPENAI_API_KEY and ensure Ollama is running with llama2:chat pulled.")
    #     return

    # config_manager = LLMConfigManager(config_file_path=str(config_path))
    # manager = LLMManager(config_manager)

    # print("Available models:", manager.list_available_models())
    # print("Active model:", manager.get_active_model_id())

    # if manager.get_active_model_id():
    #     try:
    #         response = await manager.generate_response("Hello, who are you?")
    #         print(f"Response from active model ({manager.get_active_model_id()}): {response}")
    #     except Exception as e:
    #         print(f"Error generating response: {e}")

    # # Example of switching model (if another model is configured and loaded)
    # # target_model = "ollama_llama2_chat" # Or whatever key is in your config
    # # if target_model in manager.list_available_models():
    # #    manager.set_active_model(target_model)
    # #    print(f"Switched active model to: {manager.get_active_model_id()}")
    # #    response = await manager.generate_response("Tell me a joke about computers.")
    # #    print(f"Response from {manager.get_active_model_id()}: {response}")

    # await manager.close_all_services()

# if __name__ == "__main__":
#     import asyncio
#     # Ensure OPENAI_API_KEY is set in your environment if testing with OpenAI
#     # Ensure Ollama server is running and models are pulled if testing with Ollama
#     asyncio.run(_example_manager_usage())
