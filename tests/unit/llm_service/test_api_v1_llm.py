import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

# Module to be tested
from llm_service.api_v1_llm import router as llm_api_router
from llm_service.api_v1_llm import LLMTestPromptRequest, LLMTestPromptResponse # Renamed Pydantic models

# Dependencies to be mocked
from llm_service.manager import LLMManager
from llm_service.prompts import PromptTemplateManager
from llm_service.base import LLMContext, AbstractLLMService
from llm_service.exceptions import LLMModelNotFoundError, PromptTemplateNotFoundError, PromptTemplateError, LLMServiceError


# --- Test Setup: Create a FastAPI app and TestClient ---

# Create mock instances for dependencies
mock_llm_manager = MagicMock(spec=LLMManager)
mock_prompt_template_manager = MagicMock(spec=PromptTemplateManager)

# Dependency override functions
def override_get_llm_manager():
    return mock_llm_manager

def override_get_prompt_template_manager():
    return mock_prompt_template_manager

# Create a test FastAPI app instance
app = FastAPI()
app.include_router(llm_api_router, prefix="/api/v1") # Prefix matches usage in main.py

# Apply dependency overrides to the test app
# Note: The actual dependency functions (get_llm_manager, get_prompt_template_manager)
# are assumed to be discoverable by FastAPI in the main app. Here, we override them
# for the router attached to our test app.
# The endpoint uses `Depends()` without arguments, relying on type hints or names.
# FastAPI's `dependency_overrides` works by matching the dependency callable.
# Since the endpoint's Depends() are placeholders, we need to ensure they are resolvable
# to something that can be overridden. For this test setup, we'll override based on the
# types expected by Depends(), assuming FastAPI resolves them this way in the test app.
# This is a bit indirect. A more direct way is if the endpoint explicitly used:
# `Depends(get_llm_manager_dependency_from_main_or_shared_location)`
# For now, this setup implies we are overriding what FastAPI would resolve for
# `Depends(LLMManager)` and `Depends(PromptTemplateManager)`.
# The actual functions `get_llm_manager` from `main.py` are not directly imported here.
# Instead, we override the dependencies that the router *expects* to be provided by the app.

# The endpoint signature is:
# async def test_llm_prompt(
# req_data: TestPromptRequest,
# llm_manager: LLMManager = Depends(),
# prompt_template_manager: PromptTemplateManager = Depends()
# ):
# Import the placeholder functions from the router's module to override them
from llm_service.api_v1_llm import get_llm_manager_placeholder, get_prompt_template_manager_placeholder

app.dependency_overrides[get_llm_manager_placeholder] = override_get_llm_manager
app.dependency_overrides[get_prompt_template_manager_placeholder] = override_get_prompt_template_manager


client = TestClient(app)


# --- Helper to create mock service for LLMManager ---
def create_mock_llm_service(service_name="mock_llm_service", response_text="mocked response"):
    service = MagicMock(spec=AbstractLLMService)
    service.get_service_name.return_value = service_name
    service.generate_response = AsyncMock(return_value=response_text)
    return service

# --- Test Cases ---

class TestLLMTestPromptEndpoint:

    def setup_method(self):
        # Reset mocks before each test
        mock_llm_manager.reset_mock()
        mock_prompt_template_manager.reset_mock()

        # Explicitly clear side_effect that might persist on the get_service mock itself
        # if it was set in a previous test. reset_mock() on the parent should do this,
        # but this ensures a clean state for return_value to take precedence.
        mock_llm_manager.get_service.side_effect = None

        # Default behavior for mocks (can be overridden in specific tests)
        self.mock_default_service = create_mock_llm_service(service_name="default_mock_service", response_text="default_response")
        mock_llm_manager.get_service.return_value = self.mock_default_service
        mock_prompt_template_manager.format_template.return_value = "formatted prompt"


    def test_successful_basic_prompt(self):
        request_data = LLMTestPromptRequest(prompt="Hello world") # Renamed

        # Ensure get_service returns the default mock for this test
        mock_llm_manager.get_service.return_value = self.mock_default_service

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["response"] == "default_response"
        assert response_json["model_used"] == "default_mock_service"

        mock_llm_manager.get_service.assert_called_once_with(None) # Default model
        self.mock_default_service.generate_response.assert_awaited_once()
        # Check context and prompt passed to service.generate_response
        call_args = self.mock_default_service.generate_response.call_args
        assert call_args.kwargs['prompt'] == "Hello world" # Access via kwargs
        assert isinstance(call_args.kwargs['context'], LLMContext)
        # The endpoint creates a new LLMContext, adds history, then the service adds the current prompt.
        # The context *sent* to generate_response will not have the current prompt yet.
        assert call_args.kwargs['context'].get_history() == []
        assert call_args.kwargs['settings'] == {}

        mock_prompt_template_manager.format_template.assert_not_called()

    def test_successful_prompt_with_model_id_and_template(self):
        specific_service_mock = create_mock_llm_service(service_name="specific_mock_service", response_text="specific_response")
        mock_llm_manager.get_service.return_value = specific_service_mock

        request_data = LLMTestPromptRequest( # Renamed
            prompt="User question: {user_query}",
            model_id="specific_model_id",
            template_name="my_template",
            template_params={"user_query": "What is FastAPI?"}
        )

        formatted_prompt_text = "Formatted user question: What is FastAPI?"
        mock_prompt_template_manager.format_template.return_value = formatted_prompt_text

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        assert response.status_code == 200
        response_json = response.json()
        assert response_json["response"] == "specific_response"
        assert response_json["model_used"] == "specific_mock_service"

        mock_llm_manager.get_service.assert_called_once_with("specific_model_id")
        mock_prompt_template_manager.format_template.assert_called_once_with(
            "my_template", **request_data.template_params
        )
        specific_service_mock.generate_response.assert_awaited_once()
        call_args = specific_service_mock.generate_response.call_args
        assert call_args.kwargs['prompt'] == formatted_prompt_text # Access via kwargs

    def test_successful_prompt_with_context_history(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        request_data = LLMTestPromptRequest(prompt="New question", context_history=history) # Renamed

        # Ensure get_service returns the default mock for this test
        mock_llm_manager.get_service.return_value = self.mock_default_service

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        assert response.status_code == 200
        self.mock_default_service.generate_response.assert_awaited_once()
        call_args = self.mock_default_service.generate_response.call_args
        passed_context: LLMContext = call_args.kwargs['context'] # Access via kwargs

        # History provided to service's generate_response
        expected_history_for_service_method = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"}
        ]
        # Full history including the latest prompt for the response model
        expected_history_in_response = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
            {"role": "user", "content": "New question"}
        ]
        assert passed_context.get_history() == expected_history_for_service_method
        assert response.json()["context_sent"] == expected_history_in_response


    def test_error_model_not_found(self):
        mock_llm_manager.get_service.side_effect = LLMModelNotFoundError(model_id="unknown_model")
        request_data = LLMTestPromptRequest(prompt="test", model_id="unknown_model") # Renamed

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        assert response.status_code == status.HTTP_404_NOT_FOUND
        # Check specific message content rather than class name in string
        # Actual message: "LLM model with ID 'unknown_model' not found or could not be loaded. (Model ID: unknown_model)"
        assert "LLM model with ID 'unknown_model' not found or could not be loaded." in response.json()["detail"]
        assert "(Model ID: unknown_model)" in response.json()["detail"]

    def test_error_template_not_found(self):
        mock_prompt_template_manager.format_template.side_effect = PromptTemplateNotFoundError(template_name="unknown_template")
        request_data = LLMTestPromptRequest(prompt="test", template_name="unknown_template") # Renamed

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "Prompt template 'unknown_template' not found" in response.json()["detail"]

    def test_error_template_formatting_key_error(self):
        # This simulates a KeyError during template.format(**kwargs) inside the manager
        mock_prompt_template_manager.format_template.side_effect = PromptTemplateError("Missing value for placeholder 'query'")
        request_data = LLMTestPromptRequest(prompt="test {query}", template_name="my_template", template_params={}) # Renamed

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        # The endpoint catches generic Exception for this and returns 500
        # Or specific PromptTemplateError if handled explicitly
        # Current api_v1_llm.py catches PromptTemplateError as 404, but this is a formatting error, not found.
        # Let's assume 400 or 500 is more appropriate. The current code will raise 404.
        # Let's refine the endpoint to catch PromptTemplateError specifically for formatting vs not found.
        # For now, I'll test against current behavior (which might be a 404 due to broad catch)
        # The actual exception LLMResponseError will be raised, and caught by FastAPI's default 500 handler
        # unless the endpoint has more specific handling.
        # The `api_v1_llm.py` catches `PromptTemplateNotFoundError` as 404.
        # Other `PromptTemplateError` (like formatting) would fall into generic `Exception` -> 500.
        # Or, if the `PromptTemplateError` inherits from something FastAPI handles differently.
        # Let's test for what the endpoint *should* do for a bad request due to formatting.
        # A 400 Bad Request would be appropriate if the endpoint caught PromptTemplateError.
        # If not caught specifically, it becomes a 500.
        # The current code has: except PromptTemplateNotFoundError as e: -> 404
        # It does not have a specific catch for PromptTemplateError (the base class for formatting issues)
        # So it will fall to the generic Exception e: -> 500
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # The endpoint's generic exception handler prefixes "An unexpected error occurred: "
        # or "LLM Service Error: " if it's an LLMServiceError specifically
        assert "LLM Service Error: Missing value for placeholder 'query'" in response.json()["detail"]


    def test_error_llm_service_error(self):
        # Ensure get_service returns the default mock for this test
        mock_llm_manager.get_service.return_value = self.mock_default_service
        unique_error_message = "UNIQUE_LLM_SERVICE_ERROR_FOR_TESTING_500_PATH"
        self.mock_default_service.generate_response.side_effect = LLMServiceError(unique_error_message)
        request_data = LLMTestPromptRequest(prompt="test") # Renamed

        response = client.post("/api/v1/test-prompt", json=request_data.model_dump())

        if response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR:
            print(f"DEBUG: Unexpected response status: {response.status_code}")
            try:
                print(f"DEBUG: Unexpected response detail: {response.json()['detail']}")
            except:
                print(f"DEBUG: Unexpected response content: {response.text}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert f"LLM Service Error: {unique_error_message}" in response.json()["detail"]

    def test_invalid_request_body_missing_prompt(self):
        # 'prompt' is a required field in LLMTestPromptRequest
        invalid_payload = {"model_id": "test"}
        response = client.post("/api/v1/test-prompt", json=invalid_payload)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY # FastAPI's default for Pydantic validation errors
