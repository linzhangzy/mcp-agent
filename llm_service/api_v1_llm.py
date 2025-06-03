from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

# Relative imports for components within llm_service
from .manager import LLMManager
from .prompts import PromptTemplateManager
from .base import LLMContext
from .exceptions import LLMModelNotFoundError, PromptTemplateNotFoundError, LLMServiceError

# Assuming get_llm_manager and get_prompt_template_manager are defined in main.py
# For direct invocation or testing of this router, those dependencies would need to be available
# or mocked. In a real app, they are provided by the FastAPI app instance.
# For now, we'll assume they are correctly set up in the main app.
# If main.py is at root, and this is in llm_service/, need to adjust import for dependencies:
# from main import get_llm_manager, get_prompt_template_manager
# This creates a circular dependency if main directly imports this.
# A common pattern is to have dependencies in a separate 'dependencies.py' file
# or have them passed during router inclusion.
# For this task, we assume FastAPI handles dependency resolution from `main.py` correctly.

# Re-importing dependencies from main.py for clarity if this file were run standalone (it won't be)
# This is illustrative; FastAPI uses the app's dependency system.
# from main import get_llm_manager, get_prompt_template_manager # This line is conceptual

router = APIRouter()

# --- Placeholder Dependencies for a standalone router context / easier testing ---
# These would be overridden by the main application's dependency system.
# In a main app, these could point to functions in main.py or a shared dependency module.

async def get_llm_manager_placeholder() -> LLMManager:
    # This placeholder will be overridden in tests and by the main app.
    # It's here to make the router self-contained for definition, if not for execution.
    raise NotImplementedError("LLMManager dependency not provided.")

async def get_prompt_template_manager_placeholder() -> PromptTemplateManager:
    raise NotImplementedError("PromptTemplateManager dependency not provided.")

# --- Pydantic Models for Request/Response ---

class LLMTestPromptRequest(BaseModel): # Renamed
    prompt: str
    model_id: Optional[str] = None
    template_name: Optional[str] = None
    template_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    context_history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    # Example context_history: [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]
    settings: Optional[Dict[str, Any]] = Field(default_factory=dict) # For LLM-specific settings

class LLMTestPromptResponse(BaseModel): # Renamed
    model_used: str
    response: str
    context_sent: List[Dict[str, str]] # The full context sent to the LLM

class ErrorResponse(BaseModel):
    detail: str

# --- API Endpoint ---

@router.post(
    "/test-prompt",
    response_model=LLMTestPromptResponse, # Renamed
    responses={
        404: {"model": ErrorResponse, "description": "Model or Template not found"},
        500: {"model": ErrorResponse, "description": "LLM Service error"},
        503: {"model": ErrorResponse, "description": "LLM Manager or Template Manager not available"}
    }
)
async def test_llm_prompt(
    request_data: LLMTestPromptRequest, # Renamed
    llm_manager: LLMManager = Depends(get_llm_manager_placeholder),
    prompt_template_manager: PromptTemplateManager = Depends(get_prompt_template_manager_placeholder)
):
    """
    Tests a prompt with an LLM, optionally using a template and context history.
    """
    # This is where we'd actually get the dependencies from the main app if this file was separate
    # For now, we assume they are injected by FastAPI when the router is included in main.py
    # So, we need to define these dependencies in main.py and ensure they are passed to the router.
    # The Depends() without arguments will rely on FastAPI to find functions named
    # get_llm_manager and get_prompt_template_manager if they are not overridden at router inclusion.
    # Let's make sure main.py defines these and we refer to them correctly.
    # For this task, I'll assume they are defined in main and use them.
    # from main import get_llm_manager as actual_get_llm_manager
    # from main import get_prompt_template_manager as actual_get_pt_manager
    # llm_manager = actual_get_llm_manager() # This is not how Depends() works, it calls the function.
    # The Depends() will call the functions defined in main.py.

    final_prompt = request_data.prompt
    target_model_id = request_data.model_id # If None, LLMManager uses its active model

    try:
        # 1. Handle prompt templating
        if request_data.template_name:
            final_prompt = prompt_template_manager.format_template(
                request_data.template_name,
                **(request_data.template_params or {})
            )

        # 2. Set up LLMContext
        # For this test endpoint, we'll use the default max_history_turns from LLMContext itself if any.
        # A more advanced setup might pass this from config or request.
        llm_context = LLMContext() # No system prompt for this simple test unless part of template

        if request_data.context_history:
            for message in request_data.context_history:
                # Basic validation for role, actual LLMContext add_message handles stricter validation
                if "role" in message and "content" in message:
                    if message["role"] == "system" and not llm_context.initial_system_prompt:
                        # If a system message is in history, and context doesn't have one, set it.
                        # This is a bit of a workaround for this test endpoint.
                        # Better: LLMContext handles system prompt in constructor primarily.
                        llm_context.initial_system_prompt = message["content"]
                    elif message["role"] in ["user", "assistant"]:
                         llm_context.add_message(message["role"], message["content"])


        # The final "prompt" from request is added as the last user message
        # (LLMContext's add_user_message will then add it to its internal history)
        # This is done by the llm_manager.generate_response or service.generate_response

        # 3. Get the LLM service and generate response
        # LLMManager's generate_response method handles getting the service.
        service_to_use = llm_manager.get_service(target_model_id)
        if not service_to_use:
            actual_model_id_used = target_model_id or llm_manager.get_active_model_id() or "unknown"
            raise LLMModelNotFoundError(model_id=actual_model_id_used, message=f"LLM service for model ID '{actual_model_id_used}' could not be found or is not loaded.")

        # Add the final_prompt to the context (generate_response expects the prompt separately)
        # The service's generate_response will then typically add this as the last user message.

        llm_response_text = await service_to_use.generate_response(
            prompt=final_prompt,
            context=llm_context,
            settings=request_data.settings
        )

        # Construct context sent for response
        # Add the final user prompt to context for accurate representation of what was "sent"
        # (even though service.generate_response takes it as a separate arg)
        temp_final_context = llm_context.get_history()
        temp_final_context.append({"role": "user", "content": final_prompt})


        return LLMTestPromptResponse( # Renamed
            model_used=service_to_use.get_service_name(), # Get the actual service name
            response=llm_response_text,
            context_sent=temp_final_context
        )

    except LLMModelNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except PromptTemplateNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except LLMServiceError as e: # Catch-all for other LLM related errors from services
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"LLM Service Error: {str(e)}")
    except HTTPException: # Re-raise if it's already an HTTPException (e.g. from dependencies)
        raise
    except Exception as e: # Catch any other unexpected errors
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An unexpected error occurred: {str(e)}")

# To make Depends() work without explicit provision when router is included:
# The functions get_llm_manager and get_prompt_template_manager must be resolvable by FastAPI.
# This usually means they are defined in the same file or imported into the main app module.
# The current setup in main.py (global functions) should be fine.
# If we had:
# from main import get_llm_manager, get_prompt_template_manager
# @router.post("/test-prompt")
# async def test_llm_prompt(
# req_data: TestPromptRequest,
# llm_manager: LLMManager = Depends(get_llm_manager), # Explicitly passing
# ...
# ): ...
# This would also work and is more explicit. The `Depends()` without args works if the dependency function
# has the same name as the parameter type hint, or if a default is provided.
# Here, we rely on FastAPI's default behavior with type hints.
# The `Depends()` in the endpoint signature should correctly use the dependencies from `main.py`.
