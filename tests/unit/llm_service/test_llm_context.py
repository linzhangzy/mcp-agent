import pytest
import logging # Import logging
from llm_service.base import LLMContext # Assuming llm_service is in PYTHONPATH

# Configure logging for tests (optional, but can be helpful)
logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.DEBUG) # Uncomment for verbose test logging

class TestLLMContext:

    def test_initialization_defaults(self):
        """Test LLMContext initialization with default parameters."""
        context = LLMContext()
        assert context.initial_system_prompt is None
        assert context.history == []
        assert context.max_history_turns is None
        assert context.get_history() == []

    def test_initialization_with_system_prompt(self):
        """Test LLMContext initialization with a system prompt."""
        system_prompt = "You are a helpful assistant."
        context = LLMContext(system_prompt=system_prompt)
        assert context.initial_system_prompt == system_prompt
        assert context.history == []
        expected_history = [{"role": "system", "content": system_prompt}]
        assert context.get_history() == expected_history

    def test_initialization_with_max_history_turns(self):
        """Test LLMContext initialization with max_history_turns."""
        context = LLMContext(max_history_turns=5)
        assert context.max_history_turns == 5

    def test_initialization_with_invalid_max_history_turns(self):
        """Test LLMContext initialization with zero or negative max_history_turns."""
        context_zero = LLMContext(max_history_turns=0)
        assert context_zero.max_history_turns is None # Should be reset to None

        context_negative = LLMContext(max_history_turns=-5)
        assert context_negative.max_history_turns is None # Should be reset to None

    def test_add_user_and_assistant_messages(self):
        """Test adding user and assistant messages."""
        context = LLMContext()
        context.add_user_message("Hello, AI!")
        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "Hello, AI!"}

        context.add_assistant_message("Hello, User!")
        assert len(context.history) == 2
        assert context.history[1] == {"role": "assistant", "content": "Hello, User!"}

        expected_full_history = [
            {"role": "user", "content": "Hello, AI!"},
            {"role": "assistant", "content": "Hello, User!"}
        ]
        assert context.get_history() == expected_full_history

    def test_add_message_invalid_role(self):
        """Test adding a message with an invalid role."""
        context = LLMContext()
        # Match the new error message format which uses a sorted list string representation
        expected_match_str = "Invalid role 'system_user'. Must be one of \\['assistant', 'user'\\]\\."
        with pytest.raises(ValueError, match=expected_match_str):
            context.add_message("system_user", "Trying to sneak in.")

        expected_match_system_str = "Invalid role 'system'. Must be one of \\['assistant', 'user'\\]\\."
        with pytest.raises(ValueError, match=expected_match_system_str):
            # System messages should not be added via add_message
            context.add_message("system", "This is a system message.")


    def test_get_history_prepends_system_prompt(self):
        """Test that get_history correctly prepends the system prompt."""
        system_prompt = "Be concise."
        context = LLMContext(system_prompt=system_prompt)
        context.add_user_message("Tell me a story.")

        full_history = context.get_history()
        assert len(full_history) == 2
        assert full_history[0] == {"role": "system", "content": system_prompt}
        assert full_history[1] == {"role": "user", "content": "Tell me a story."}

    def test_get_history_returns_copy(self):
        """Test that get_history returns a copy of the history, not the internal list."""
        context = LLMContext()
        context.add_user_message("Original message.")

        history_copy1 = context.get_history()
        assert len(history_copy1) == 1

        history_copy1.append({"role": "user", "content": "Modified copy."})

        # Original history should be unchanged
        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "Original message."}
        assert len(context.get_history()) == 1


    def test_clear_history(self):
        """Test clearing the history."""
        system_prompt = "Always be polite."
        context = LLMContext(system_prompt=system_prompt, max_history_turns=5)
        context.add_user_message("Question 1")
        context.add_assistant_message("Answer 1")

        assert len(context.history) == 2

        context.clear_history()
        assert len(context.history) == 0 # User/assistant messages cleared
        assert context.initial_system_prompt == system_prompt # System prompt retained

        # get_history should now only contain the system prompt
        expected_history_after_clear = [{"role": "system", "content": system_prompt}]
        assert context.get_history() == expected_history_after_clear

    # --- Tests for Truncation ---

    @pytest.mark.parametrize("max_turns", [1, 2, 3])
    def test_truncation_simple(self, max_turns):
        """Test basic truncation behavior."""
        context = LLMContext(max_history_turns=max_turns)

        for i in range(max_turns + 2): # Add more turns than allowed
            context.add_user_message(f"User message {i+1}")
            context.add_assistant_message(f"Assistant message {i+1}")

        # History should contain `max_turns` user messages and `max_turns` assistant messages
        # Total messages in self.history should be max_turns * 2
        assert len(context.history) == max_turns * 2

        # Verify that the correct messages were kept (the most recent ones)
        user_messages_in_history = [msg for msg in context.history if msg["role"] == "user"]
        assert len(user_messages_in_history) == max_turns

        # Check content of the first user message (should be the (total_user_messages - max_turns + 1)th message)
        # Example: 5 turns added, max_turns=2. Kept: U4, A4, U5, A5. First user message is U4. Original index of U4 was 3.
        # (max_turns + 2) - max_turns = 2. So, User message 3 (index 2) should be the first.
        # The first user message content should be "User message { (max_turns + 2) - max_turns + 1 }"
        # No, it should be: total_user_messages_added = max_turns + 2.
        # First user message to be kept is user_message_index = (total_user_messages_added - max_turns)
        # So content is "User message { (max_turns + 2) - max_turns + 1 }" = "User message 3" if max_turns = 2, total added = 4
        # total_user_messages_added = max_turns + 2
        # first_kept_user_message_number = (max_turns + 2) - max_turns + 1 = 3
        first_kept_user_message_number = (max_turns + 2) - max_turns + 1
        assert context.history[0]["content"] == f"User message {first_kept_user_message_number}"
        assert context.history[1]["content"] == f"Assistant message {first_kept_user_message_number}"


    def test_truncation_with_system_prompt(self):
        """Test truncation when a system prompt is present."""
        system_prompt = "System instructions."
        max_turns = 2
        context = LLMContext(system_prompt=system_prompt, max_history_turns=max_turns)

        for i in range(max_turns + 2): # Add 4 turns
            context.add_user_message(f"U{i+1}")
            context.add_assistant_message(f"A{i+1}")

        full_history = context.get_history()
        # Expected: System, U3, A3, U4, A4
        assert len(full_history) == 1 + (max_turns * 2)
        assert full_history[0]["role"] == "system"
        assert full_history[0]["content"] == system_prompt

        assert full_history[1]["content"] == "U3" # (max_turns + 2) - max_turns + 1 = 4 - 2 + 1 = 3
        assert full_history[2]["content"] == "A3"

    def test_truncation_no_assistant_after_last_user(self):
        """Test truncation when the last message is a user message."""
        max_turns = 1
        context = LLMContext(max_history_turns=max_turns)
        context.add_user_message("U1")
        context.add_assistant_message("A1")
        context.add_user_message("U2") # This should trigger truncation, U1/A1 removed

        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "U2"}

    def test_truncation_multiple_assistants_after_user(self):
        """Test truncation when multiple assistant messages follow a user message (though atypical)."""
        # Current _truncate_history removes the user message and subsequent assistant messages
        # until the next user message or end of history.
        max_turns = 1
        context = LLMContext(max_history_turns=max_turns)
        context.add_user_message("U1")
        context.add_assistant_message("A1.1")
        context.add_assistant_message("A1.2") # These two are a "turn" with U1
        context.add_user_message("U2")        # Triggers truncation of U1, A1.1, A1.2

        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "U2"}

        context.add_assistant_message("A2.1")
        context.add_user_message("U3") # Triggers truncation of U2, A2.1
        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "U3"}


    def test_no_truncation_if_max_turns_not_set(self):
        """Test that no truncation occurs if max_history_turns is None."""
        context = LLMContext(max_history_turns=None)
        for i in range(5):
            context.add_user_message(f"U{i+1}")
            context.add_assistant_message(f"A{i+1}")
        assert len(context.history) == 10 # All messages retained

    def test_no_truncation_if_below_max_turns(self):
        """Test no truncation occurs if history length is below max_history_turns."""
        max_turns = 3
        context = LLMContext(max_history_turns=max_turns)
        context.add_user_message("U1")
        context.add_assistant_message("A1")
        context.add_user_message("U2") # 2 user turns, less than 3

        assert len(context.history) == 3 # U1, A1, U2
        assert context.history[0]["content"] == "U1"
        assert context.history[1]["content"] == "A1"
        assert context.history[2]["content"] == "U2"

    def test_truncation_exact_max_turns(self):
        """Test behavior when number of turns equals max_history_turns."""
        max_turns = 2
        context = LLMContext(max_history_turns=max_turns)
        context.add_user_message("U1")
        context.add_assistant_message("A1")
        context.add_user_message("U2") # Triggers no truncation yet
        context.add_assistant_message("A2")

        assert len(context.history) == 4 # U1, A1, U2, A2

        context.add_user_message("U3") # Triggers truncation of U1, A1
        assert len(context.history) == 3 # U2, A2, U3
        assert context.history[0]["content"] == "U2"
        assert context.history[1]["content"] == "A2"
        assert context.history[2]["content"] == "U3"

    def test_add_message_triggers_truncation(self):
        """Ensure add_message (not just add_user_message) also respects truncation logic if role is user."""
        max_turns = 1
        context = LLMContext(max_history_turns=max_turns)
        context.add_message("user", "U1")
        context.add_message("assistant", "A1")
        context.add_message("user", "U2") # Should truncate U1, A1

        assert len(context.history) == 1
        assert context.history[0] == {"role": "user", "content": "U2"}

    def test_repr_method(self):
        """Test the __repr__ method for LLMContext."""
        context = LLMContext()
        assert repr(context) == "<LLMContext system_prompt=False history_length=0 max_turns=None>"

        context_sys = LLMContext(system_prompt="System here", max_history_turns=3)
        context_sys.add_user_message("Hi")
        assert repr(context_sys) == "<LLMContext system_prompt=True history_length=1 max_turns=3>"
