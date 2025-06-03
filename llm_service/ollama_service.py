import httpx
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path # Added for main_ollama_test

from .base import AbstractLLMService, LLMContext
from .config import LLMConfigManager, LLMModelConfig, LLMModelSettings
from .exceptions import LLMServiceError, LLMConfigError, LLMResponseError

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_TIMEOUT = 60.0  # seconds, potentially longer for local models
DEFAULT_KEEP_ALIVE = "5m" # Default keep_alive for Ollama models

class OllamaService(AbstractLLMService):
    """
    An implementation of AbstractLLMService for local Ollama models
    using Ollama's REST API (primarily /api/chat).
    """

    def __init__(self, model_id: str, config_manager: LLMConfigManager):
        """
        Initializes the OllamaService.

        Args:
            model_id: The key for the specific Ollama model configuration
                      (e.g., "llama3_8b", "codellama_7b").
            config_manager: An instance of LLMConfigManager that has loaded
                            the LLM configurations.

        Raises:
            LLMConfigError: If the specified model_id or provider configuration
                            is not found, or if the base_url is missing.
        """
        provider_config = config_manager.get_provider_config("ollama")
        if not provider_config:
            raise LLMConfigError("Ollama provider configuration not found.")

        model_config = config_manager.get_model_config("ollama", model_id)
        if not model_config:
            raise LLMConfigError(f"Configuration for Ollama model ID '{model_id}' not found.")

        super().__init__(model_name=model_config.model_name, config=model_config.model_dump(exclude_none=True))

        self.ollama_model_name: str = model_config.model_name # e.g., "llama3:8b"
        self.base_url = (provider_config.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip('/')
        self.api_endpoint = self.base_url + "/api/chat"

        # Merge default settings: provider-level defaults < model-level defaults
        provider_default_settings = LLMModelSettings(keep_alive=provider_config.default_keep_alive or DEFAULT_KEEP_ALIVE)
        model_default_settings = model_config.default_settings or LLMModelSettings()

        # Pydantic model_dump will exclude None values, then we update
        merged_defaults = provider_default_settings.model_dump(exclude_none=True)
        merged_defaults.update(model_default_settings.model_dump(exclude_none=True))
        self.default_settings = LLMModelSettings(**merged_defaults)

        self.client = httpx.AsyncClient(timeout=config_manager.get_global_timeout() or DEFAULT_TIMEOUT)
        self._service_name_override = f"ollama_{model_id}"


    def get_service_name(self) -> str:
        """Returns a unique identifier for this service configuration."""
        return self._service_name_override

    async def generate_response(
        self,
        prompt: str,
        context: Optional[LLMContext] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a response from the Ollama /api/chat endpoint.

        Args:
            prompt: The user's current prompt or query.
            context: Optional LLMContext object containing conversation history.
            settings: Optional dictionary of Ollama-specific runtime settings.
                      These are passed in the 'options' field of the request.
                      The 'keep_alive' setting can also be passed here.

        Returns:
            The LLM's response content as a string.

        Raises:
            LLMResponseError: If the API call fails, returns an error, or the response
                              format is unexpected.
        """
        messages: List[Dict[str, str]] = []
        if context:
            messages.extend(context.get_history())
        messages.append({"role": "user", "content": prompt})

        # Prepare combined settings for 'options' and 'keep_alive'
        final_options = self.default_settings.model_dump(exclude_none=True)

        # Separate keep_alive from other settings for payload construction
        final_keep_alive = final_options.pop("keep_alive", self.default_settings.keep_alive or DEFAULT_KEEP_ALIVE)

        if settings:
            runtime_keep_alive = settings.pop("keep_alive", None)
            if runtime_keep_alive is not None:
                final_keep_alive = runtime_keep_alive
            final_options.update(settings)


        payload = {
            "model": self.ollama_model_name,
            "messages": messages,
            "options": final_options,
            "stream": False, # Explicitly set to False to get a single JSON response
            "keep_alive": final_keep_alive
        }

        headers = {"Content-Type": "application/json"}

        try:
            response = await self.client.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

            response_data = response.json()

            # For non-streaming chat, the response structure is:
            # { "model": "...", "created_at": "...", "message": {"role": "assistant", "content": "..." }, "done": true, ... }
            if not response_data.get("message") or "content" not in response_data["message"]:
                error_detail = response_data.get("error", "Invalid response structure from Ollama API: 'message.content' missing.")
                raise LLMResponseError(
                    error_detail,
                    provider_name=self.get_service_name()
                )

            assistant_response = response_data["message"]["content"]
            return assistant_response.strip()

        except httpx.HTTPStatusError as e:
            error_details = "No error details in response."
            try:
                # Ollama errors are often plain text or have an 'error' key in JSON
                ollama_error = e.response.json()
                if "error" in ollama_error:
                    error_details = ollama_error["error"]
                else:
                    error_details = e.response.text
            except json.JSONDecodeError:
                error_details = e.response.text # Fallback to raw text if not JSON
            raise LLMResponseError(
                f"Ollama API request failed with status {e.response.status_code}: {error_details}",
                provider_name=self.get_service_name(),
                status_code=e.response.status_code
            )
        except httpx.TimeoutException as e:
            raise LLMResponseError(
                f"Request to Ollama API timed out: {e}",
                provider_name=self.get_service_name()
            )
        except httpx.RequestError as e: # Covers connection errors, etc.
            raise LLMResponseError(
                f"An error occurred during the request to Ollama API: {e}",
                provider_name=self.get_service_name()
            )
        except json.JSONDecodeError as e:
             raise LLMResponseError(f"Failed to decode JSON response from Ollama API: {e}", provider_name=self.get_service_name())


    async def close(self):
        """Closes the httpx client. Should be called on application shutdown."""
        await self.client.aclose()


# Example of how to use (for testing or direct use)
async def main_ollama_test():
    # This requires an Ollama server running and the specified model pulled.
    # Create a dummy llm_config.yaml for this test
    dummy_config_content = """
default_model_id: "ollama_test_model"
llm_providers:
  ollama:
    base_url: "http://localhost:11434" # Ensure this is correct for your Ollama instance
    default_keep_alive: "5m"
    models:
      test_model: # This is the model_id used in init
        model_name: "llama3:8b" # CHANGE THIS to a model you have pulled, e.g., "llama2", "mistral"
        default_settings:
          temperature: 0.3
          num_ctx: 1024 # Small context for testing
"""
    # Note: For real usage, llm_config.yaml would be in llm_service/ or path passed to LLMConfigManager
    dummy_config_path = Path(__file__).parent / "llm_config.yaml"
    with open(dummy_config_path, 'w') as f:
        f.write(dummy_config_content)

    try:
        print(f"Attempting to load config from: {dummy_config_path.resolve()}")
        config_manager = LLMConfigManager(config_file_path=str(dummy_config_path))

        # Use the key from the YAML file, e.g., "test_model"
        ollama_service = OllamaService(model_id="test_model", config_manager=config_manager)

        print(f"Service Name: {ollama_service.get_service_name()}")
        print(f"Using model: {ollama_service.ollama_model_name} via {ollama_service.base_url}")

        prompt1 = "Why is the sky blue?"
        print(f"\nUser: {prompt1}")
        # Test with keep_alive override
        response1 = await ollama_service.generate_response(prompt1, settings={"keep_alive": "1m"})
        print(f"AI: {response1}")

        context = LLMContext()
        context.add_message("user", prompt1)
        context.add_message("assistant", response1)

        prompt2 = "And what about during sunset?"
        print(f"\nUser: {prompt2}")
        response2 = await ollama_service.generate_response(prompt2, context=context)
        print(f"AI: {response2}")

    except LLMConfigError as e:
        print(f"Configuration Error: {e}")
    except LLMResponseError as e:
        print(f"API Response Error: {e}")
    except httpx.ConnectError as e:
        print(f"Connection Error: Could not connect to Ollama server at {ollama_service.base_url if 'ollama_service' in locals() else DEFAULT_OLLAMA_BASE_URL}. Is Ollama running? Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'ollama_service' in locals() and ollama_service:
            await ollama_service.close()
        if dummy_config_path.exists():
            dummy_config_path.unlink()


if __name__ == "__main__":
    import asyncio
    # To run this test, ensure you have an Ollama server running
    # and the model specified in `dummy_config_content` (e.g., "llama3:8b") is pulled.
    # Example: `ollama pull llama3:8b`
    print("Running OllamaService main_test. Ensure Ollama is running and the model is pulled.")
    asyncio.run(main_ollama_test())
