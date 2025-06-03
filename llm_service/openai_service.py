import httpx
import json
import os
from typing import Any, Dict, List, Optional, Union

from .base import AbstractLLMService, LLMContext
from .config import LLMConfigManager, LLMModelConfig, LLMModelSettings
from .exceptions import LLMServiceError, LLMConfigError, LLMResponseError

# Default OpenAI API endpoint
OPENAI_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 30.0  # seconds

class OpenAIService(AbstractLLMService):
    """
    An implementation of AbstractLLMService for OpenAI's Chat Completions API.
    """

    def __init__(self, model_id: str, config_manager: LLMConfigManager):
        """
        Initializes the OpenAIService.

        Args:
            model_id: The key for the specific OpenAI model configuration
                      (e.g., "gpt_3_5_turbo", "gpt_4").
            config_manager: An instance of LLMConfigManager that has loaded
                            the LLM configurations.

        Raises:
            LLMConfigError: If the specified model_id or provider configuration
                            is not found, or if the API key is missing.
        """
        provider_config = config_manager.get_provider_config("openai")
        if not provider_config:
            raise LLMConfigError("OpenAI provider configuration not found.")

        model_config = config_manager.get_model_config("openai", model_id)
        if not model_config:
            raise LLMConfigError(f"Configuration for OpenAI model ID '{model_id}' not found.")

        super().__init__(model_name=model_config.model_name, config=model_config.model_dump(exclude_none=True))

        self.api_key: Optional[str] = provider_config.api_key
        if not self.api_key:
            env_var_name = provider_config.api_key_env_var
            if env_var_name:
                self.api_key = os.getenv(env_var_name)
            if not self.api_key:
                raise LLMConfigError(
                    f"OpenAI API key not found. Please set it directly or via environment variable '{env_var_name or 'OPENAI_API_KEY'}'."
                )

        self.openai_model_name: str = model_config.model_name # e.g., "gpt-3.5-turbo"
        self.default_settings: LLMModelSettings = model_config.default_settings or LLMModelSettings()

        # Determine API endpoint
        self.api_endpoint = (provider_config.base_url or OPENAI_API_BASE_URL).rstrip('/') + "/chat/completions"

        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT)
        self._service_name_override = f"openai_{model_id}" # Use the key from config for uniqueness

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
        Generates a response from the OpenAI Chat Completions API.

        Args:
            prompt: The user's current prompt or query.
            context: Optional LLMContext object containing conversation history.
            settings: Optional dictionary of LLM-specific runtime settings
                      (e.g., temperature, max_tokens). These override defaults.

        Returns:
            The LLM's response content as a string.

        Raises:
            LLMConfigError: If API key is missing.
            LLMResponseError: If the API call fails, returns an error, or the response
                              format is unexpected.
        """
        if not self.api_key:
            raise LLMConfigError("OpenAI API key is not configured.")

        messages: List[Dict[str, str]] = []

        # Add system message if defined in default or runtime settings
        # Runtime settings can override default system message
        final_settings = self.default_settings.model_dump(exclude_none=True)
        if settings:
            final_settings.update(settings)

        system_message_content = final_settings.pop("system_message", None) # Custom setting
        if system_message_content:
            messages.append({"role": "system", "content": system_message_content})

        if context:
            messages.extend(context.get_history())

        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.openai_model_name,
            "messages": messages,
            **final_settings  # Add merged settings (temperature, max_tokens, etc.)
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = await self.client.post(self.api_endpoint, headers=headers, json=payload)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses

            response_data = response.json()
            if not response_data.get("choices") or not response_data["choices"][0].get("message") \
               or "content" not in response_data["choices"][0]["message"]:
                raise LLMResponseError("Invalid response structure from OpenAI API.", provider_name=self.get_service_name())

            assistant_response = response_data["choices"][0]["message"]["content"]
            return assistant_response.strip()

        except httpx.HTTPStatusError as e:
            error_details = "No error details in response."
            try:
                error_details = e.response.json() # OpenAI usually returns JSON errors
            except json.JSONDecodeError:
                error_details = e.response.text
            raise LLMResponseError(
                f"OpenAI API request failed with status {e.response.status_code}: {error_details}",
                provider_name=self.get_service_name(),
                status_code=e.response.status_code
            )
        except httpx.TimeoutException as e:
            raise LLMResponseError(
                f"Request to OpenAI API timed out: {e}",
                provider_name=self.get_service_name()
            )
        except httpx.RequestError as e:
            raise LLMResponseError(
                f"An error occurred during the request to OpenAI API: {e}",
                provider_name=self.get_service_name()
            )
        except json.JSONDecodeError as e: # Should be caught by HTTPStatusError if API error is JSON
             raise LLMResponseError(f"Failed to decode JSON response from OpenAI API: {e}", provider_name=self.get_service_name())


    async def close(self):
        """Closes the httpx client. Should be called on application shutdown."""
        await self.client.aclose()


# Example of how to use (for testing or direct use, not part of the library's public API directly)
async def main_test():
    # This requires llm_config.yaml to be present in the same directory
    # and OPENAI_API_KEY environment variable to be set.
    # Create a dummy llm_config.yaml for this test
    dummy_config_content = """
default_model_id: "openai_gpt_3_5_turbo"
llm_providers:
  openai:
    api_key_env_var: "OPENAI_API_KEY"
    models:
      gpt_3_5_turbo:
        model_name: "gpt-3.5-turbo"
        default_settings:
          temperature: 0.7
          max_tokens: 50
          system_message: "You are a concise assistant."
"""
    dummy_config_path = Path(__file__).parent / "llm_config.yaml" # Assumes config is in same dir
    with open(dummy_config_path, 'w') as f:
        f.write(dummy_config_content)

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Skipping live API test.")
        dummy_config_path.unlink(missing_ok=True)
        return

    try:
        config_manager = LLMConfigManager(config_file_path=str(dummy_config_path))
        openai_service = OpenAIService(model_id="gpt_3_5_turbo", config_manager=config_manager)

        print(f"Service Name: {openai_service.get_service_name()}")
        print(f"Using model: {openai_service.openai_model_name}")

        # Test without context
        prompt1 = "What is the capital of France?"
        print(f"\nUser: {prompt1}")
        response1 = await openai_service.generate_response(prompt1)
        print(f"AI: {response1}")

        # Test with context
        context = LLMContext()
        context.add_message("user", prompt1)
        context.add_message("assistant", response1)

        prompt2 = "And what is its population?"
        print(f"\nUser: {prompt2}")
        response2 = await openai_service.generate_response(prompt2, context=context)
        print(f"AI: {response2}")

        # Test with runtime settings
        prompt3 = "Tell me a very short joke."
        print(f"\nUser: {prompt3}")
        response3 = await openai_service.generate_response(prompt3, settings={"temperature": 0.2, "max_tokens": 20})
        print(f"AI: {response3}")

    except LLMConfigError as e:
        print(f"Configuration Error: {e}")
    except LLMResponseError as e:
        print(f"API Response Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if 'openai_service' in locals() and openai_service:
            await openai_service.close()
        dummy_config_path.unlink(missing_ok=True) # Clean up dummy config


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_test())
