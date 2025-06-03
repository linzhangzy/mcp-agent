from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

VALID_ROLES = {"user", "assistant"} # System role is handled separately

class LLMContext:
    """
    Manages conversation history for interacting with LLMs,
    including system prompts and history truncation.
    """
    def __init__(self, system_prompt: Optional[str] = None, max_history_turns: Optional[int] = None):
        """
        Initializes LLMContext.

        Args:
            system_prompt: An optional initial system message to guide the LLM.
            max_history_turns: Optional maximum number of user/assistant turns to retain.
                               A "turn" is considered one user message. If the number of
                               user messages exceeds this, older turns are truncated.
        """
        self.initial_system_prompt: Optional[str] = system_prompt
        self.history: List[Dict[str, str]] = [] # Stores user and assistant messages
        self.max_history_turns: Optional[int] = max_history_turns

        if max_history_turns is not None and max_history_turns <= 0:
            logger.warning("max_history_turns must be a positive integer if set. Disabling turn limit.")
            self.max_history_turns = None

    def add_message(self, role: str, content: str) -> None:
        """
        Adds a message to the history and applies truncation if necessary.
        Only 'user' and 'assistant' roles are allowed here. System prompts
        are handled via initial_system_prompt.

        Args:
            role: The role of the message sender (e.g., "user", "assistant").
            content: The content of the message.

        Raises:
            ValueError: If an invalid role is provided.
        """
        if role not in VALID_ROLES:
            # Sort roles for consistent error message
            raise ValueError(f"Invalid role '{role}'. Must be one of {sorted(list(VALID_ROLES))}.")

        self.history.append({"role": role, "content": content})

        if role == "user": # Only truncate after a user message, as this signifies a "turn"
            self._truncate_history()

    def add_user_message(self, content: str) -> None:
        """Convenience method to add a user message."""
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Convenience method to add an assistant message."""
        self.add_message("assistant", content)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Returns the conversation history, including the system prompt if set.
        The returned history is a copy, so modifications to it won't affect
        the internal state of LLMContext.
        """
        full_history: List[Dict[str, str]] = []
        if self.initial_system_prompt:
            full_history.append({"role": "system", "content": self.initial_system_prompt})
        full_history.extend(self.history)
        return full_history

    def clear_history(self) -> None:
        """
        Clears the user/assistant message history. The initial system prompt is retained.
        """
        self.history = []
        logger.info("LLMContext history cleared (system prompt retained).")

    def _truncate_history(self) -> None:
        """
        Internal method to truncate the history based on max_history_turns.
        It counts user messages as turns. If the number of user messages
        exceeds max_history_turns, it removes the oldest user message and any
        immediately following assistant message(s) until the turn limit is met.
        """
        if self.max_history_turns is None or self.max_history_turns <= 0:
            return

        user_message_count = sum(1 for msg in self.history if msg["role"] == "user")

        while user_message_count > self.max_history_turns:
            if not self.history: # Should not happen if user_message_count > 0
                break

            # Find the first user message to remove
            first_user_message_index = -1
            for i, msg in enumerate(self.history):
                if msg["role"] == "user":
                    first_user_message_index = i
                    break

            if first_user_message_index == -1: # No user messages left, stop
                break

            # Determine how many messages to remove (the user message and any subsequent assistant messages
            # before the next user message or end of history)
            num_to_remove = 1 # Start with the user message
            for i in range(first_user_message_index + 1, len(self.history)):
                if self.history[i]["role"] == "assistant":
                    num_to_remove += 1
                else: # Next user message or other role encountered
                    break

            self.history = self.history[num_to_remove:]
            logger.debug(f"Truncated {num_to_remove} message(s) from the beginning of history.")
            user_message_count = sum(1 for msg in self.history if msg["role"] == "user")


    def __repr__(self) -> str:
        system_prompt_exists = bool(self.initial_system_prompt)
        return (f"<LLMContext system_prompt={system_prompt_exists} "
                f"history_length={len(self.history)} "
                f"max_turns={self.max_history_turns or 'None'}>")


class AbstractLLMService(ABC):
    """
    Abstract Base Class for LLM Service integrations.
    Defines the common interface that all specific LLM service wrappers must implement.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the LLM service.

        Args:
            model_name: The specific model name being used (e.g., "gpt-3.5-turbo").
            config: A dictionary containing configuration specific to this model/provider
                    (e.g., API key, base URL, model-specific parameters).
        """
        self.model_name = model_name
        self.config = config

    @abstractmethod
    async def generate_response(
        self,
        prompt: str,
        context: Optional[LLMContext] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a response from the LLM service.

        Args:
            prompt: The user's current prompt or query.
            context: Optional LLMContext object containing conversation history.
            settings: Optional dictionary of LLM-specific runtime settings
                      (e.g., temperature, max_tokens, streaming options).

        Returns:
            The LLM's response as a string.

        Raises:
            LLMResponseError: If the API call fails or returns an error.
            NotImplementedError: If the method is not implemented by the subclass.
        """
        pass

    @abstractmethod
    def get_service_name(self) -> str:
        """
        Returns a unique, descriptive name for this LLM service configuration.
        This could include provider and model, e.g., 'openai_gpt-3.5-turbo'.

        Returns:
            The unique name of this LLM service.
        """
        pass

    # Optional: Add other common methods if identified, e.g.:
    # @abstractmethod
    # async def get_embeddings(self, text: str) -> List[float]:
    #     """Generates embeddings for a given text."""
    #     pass

    # @abstractmethod
    # def validate_config(self, config: Dict[str, Any]) -> bool:
    #     """Validates the provided configuration for this service."""
    #     pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} service_name='{self.get_service_name()}' model='{self.model_name}'>"
