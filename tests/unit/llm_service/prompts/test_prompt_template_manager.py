import pytest
import yaml
import os
from pathlib import Path
from unittest.mock import mock_open, patch

# Adjust imports based on actual project structure
# Assuming llm_service is in PYTHONPATH and structured as llm_service.prompts.template_manager
from llm_service.prompts.template_manager import PromptTemplateManager
from llm_service.exceptions import PromptTemplateError, PromptTemplateNotFoundError

VALID_TEMPLATES_YAML = """
simple_greeting: |-
  Hello, {name}!
faq_formatter: |-
  Question: {question}
  Answer: {answer}
"""

INVALID_YAML_CONTENT = """
- item1
- item2
"""

NON_STRING_VALUE_YAML = """
greeting: Hello
details:
  count: 5 # This is not a string
"""

class TestPromptTemplateManager:

    def test_initialization_successful_load(self, tmp_path: Path):
        """Test successful initialization and template loading from a valid YAML file."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)

        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        assert "simple_greeting" in manager.templates
        assert manager.templates["simple_greeting"] == "Hello, {name}!"
        assert "faq_formatter" in manager.templates
        assert manager.list_template_names() == ["simple_greeting", "faq_formatter"]

    def test_initialization_default_path(self, tmp_path: Path, monkeypatch):
        """Test loading from default path if templates_file_path is None."""
        # Create a dummy prompts directory structure similar to the real one
        prompts_dir = tmp_path / "llm_service_prompts_dir" # Mocked base for template_manager.py
        prompts_dir.mkdir()
        default_templates_file = prompts_dir / "prompt_templates.yaml"
        default_templates_file.write_text(VALID_TEMPLATES_YAML)

        # Patch os.path.abspath to make the manager
        # think its __file__ is in prompts_dir for default path resolution.
        import llm_service.prompts.template_manager # To access its __file__ attribute

        path_to_template_manager_source_file = llm_service.prompts.template_manager.__file__
        original_os_path_abspath = os.path.abspath

        def mock_abspath_for_default_path(path_arg):
            if path_arg == path_to_template_manager_source_file:
                # This makes PromptTemplateManager think its own source file is here:
                return str(prompts_dir / "dummy_template_manager.py")
            return original_os_path_abspath(path_arg)

        monkeypatch.setattr(llm_service.prompts.template_manager.os.path, "abspath", mock_abspath_for_default_path)

        manager = PromptTemplateManager(templates_file_path=None) # Should use default logic based on mocked __file__ location

        assert "simple_greeting" in manager.templates
        assert manager.templates["simple_greeting"] == "Hello, {name}!"
        assert "faq_formatter" in manager.templates
        assert manager._templates_file_path == str(default_templates_file)


    def test_initialization_file_not_found(self, tmp_path: Path):
        """Test initialization when the template file is not found."""
        non_existent_file = tmp_path / "non_existent.yaml"
        with pytest.raises(PromptTemplateError, match=f"Prompt templates file not found: {non_existent_file}"):
            PromptTemplateManager(templates_file_path=str(non_existent_file))

    def test_initialization_invalid_yaml_content(self, tmp_path: Path):
        """Test initialization with a file containing invalid YAML."""
        templates_file = tmp_path / "invalid.yaml"
        templates_file.write_text("key: value: another_value # Invalid YAML")

        with pytest.raises(PromptTemplateError, match="Error parsing YAML"):
            PromptTemplateManager(templates_file_path=str(templates_file))

    def test_initialization_yaml_not_a_dictionary(self, tmp_path: Path):
        """Test initialization with YAML content that is not a dictionary."""
        templates_file = tmp_path / "list.yaml"
        templates_file.write_text(INVALID_YAML_CONTENT) # This YAML is a list at root

        with pytest.raises(PromptTemplateError, match="content is not a valid YAML dictionary"):
            PromptTemplateManager(templates_file_path=str(templates_file))

    def test_initialization_template_value_not_string(self, tmp_path: Path):
        """Test initialization with YAML where a template value is not a string."""
        templates_file = tmp_path / "non_string.yaml"
        templates_file.write_text(NON_STRING_VALUE_YAML)

        # The manager tries to str(value), so int 5 becomes "5"
        # If strict string type is required and conversion fails or is disallowed,
        # this test would change. Current implementation converts.
        manager = PromptTemplateManager(templates_file_path=str(templates_file))
        assert manager.templates["greeting"] == "Hello"
        assert manager.templates["details"] == "{'count': 5}" # The dict is str-converted

    def test_initialization_empty_file(self, tmp_path: Path):
        """Test initialization with an empty YAML file."""
        templates_file = tmp_path / "empty.yaml"
        templates_file.write_text("") # Empty file

        # yaml.safe_load on empty string returns None.
        # PromptTemplateManager now handles this by logging and creating an empty template dict.
        manager = PromptTemplateManager(templates_file_path=str(templates_file))
        assert manager.templates == {}
        assert manager.list_template_names() == []

    def test_get_template_success(self, tmp_path: Path):
        """Test retrieving an existing template."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        template_content = manager.get_template("simple_greeting")
        assert template_content == "Hello, {name}!"

    def test_get_template_not_found(self, tmp_path: Path):
        """Test retrieving a non-existent template."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        with pytest.raises(PromptTemplateNotFoundError, match="Prompt template 'non_existent_template' not found"):
            manager.get_template("non_existent_template")

    def test_format_template_success(self, tmp_path: Path):
        """Test successful formatting of a template."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        formatted = manager.format_template("simple_greeting", name="World")
        assert formatted == "Hello, World!"

        formatted_faq = manager.format_template("faq_formatter", question="What is Pytest?", answer="A testing framework.")
        assert formatted_faq == "Question: What is Pytest?\nAnswer: A testing framework."

    def test_format_template_extra_kwargs(self, tmp_path: Path):
        """Test formatting with extra keyword arguments (should be ignored by str.format)."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        formatted = manager.format_template("simple_greeting", name="Galaxy", extra_arg="ignored")
        assert formatted == "Hello, Galaxy!"

    def test_format_template_missing_placeholder(self, tmp_path: Path):
        """Test formatting when a required placeholder is missing."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        with pytest.raises(PromptTemplateError, match="Missing value for placeholder 'name' in template 'simple_greeting'"):
            manager.format_template("simple_greeting", wrong_arg="World") # 'name' is missing

    def test_format_template_template_not_found(self, tmp_path: Path):
        """Test formatting a non-existent template."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        with pytest.raises(PromptTemplateNotFoundError):
            manager.format_template("unknown_template", name="Test")

    def test_list_template_names_success(self, tmp_path: Path):
        """Test listing template names."""
        templates_file = tmp_path / "templates.yaml"
        templates_file.write_text(VALID_TEMPLATES_YAML)
        manager = PromptTemplateManager(templates_file_path=str(templates_file))

        names = manager.list_template_names()
        assert isinstance(names, list)
        assert sorted(names) == sorted(["simple_greeting", "faq_formatter"])

    def test_list_template_names_empty(self, tmp_path: Path):
        """Test listing template names when no templates are loaded (e.g. file was valid but empty dict)."""
        templates_file = tmp_path / "empty_dict.yaml"
        templates_file.write_text("{}") # Valid YAML, but an empty dictionary
        manager = PromptTemplateManager(templates_file_path=str(templates_file))
        assert manager.list_template_names() == []

    def test_load_templates_from_file_updates_path_and_templates(self, tmp_path: Path):
        """Test that load_templates_from_file correctly updates internal state."""
        initial_templates_file = tmp_path / "initial.yaml"
        initial_templates_file.write_text("initial: Hello {name}")
        manager = PromptTemplateManager(templates_file_path=str(initial_templates_file))
        assert manager.get_template("initial") == "Hello {name}"
        assert manager._templates_file_path == str(initial_templates_file)

        secondary_templates_file = tmp_path / "secondary.yaml"
        secondary_templates_file.write_text("secondary: Goodbye {name}")

        manager.load_templates_from_file(str(secondary_templates_file))
        assert "secondary" in manager.templates
        assert manager.get_template("secondary") == "Goodbye {name}"
        # Check if initial templates are cleared or merged (current implementation clears)
        assert "initial" not in manager.templates
        assert manager._templates_file_path == str(secondary_templates_file)

    def test_init_failure_if_custom_path_fails(self, tmp_path: Path):
        """Test that if a custom path is provided to __init__ and fails, it raises."""
        non_existent_custom_path = tmp_path / "custom_non_existent.yaml"
        with pytest.raises(PromptTemplateError, match=f"Prompt templates file not found: {non_existent_custom_path}"):
            PromptTemplateManager(templates_file_path=str(non_existent_custom_path))

    def test_init_graceful_if_default_path_fails(self, monkeypatch):
        """Test that if default path loading fails in __init__, it starts empty but doesn't hard fail."""
        # This test assumes that if templates_file_path is None, a failure to load the default
        # file (e.g., FileNotFoundError) results in an empty manager, rather than __init__ raising.
        # The PromptTemplateManager code has a check:
        # `if templates_file_path is not None: raise`

        # Mock abspath to point to a non-existent default file scenario
        import llm_service.prompts.template_manager # To access its __file__ attribute

        path_to_template_manager_source_file = llm_service.prompts.template_manager.__file__
        original_os_path_abspath = os.path.abspath

        def mock_abspath_for_nonexistent_default(path_arg):
            if path_arg == path_to_template_manager_source_file:
                # Make PromptTemplateManager think its source file is in a place
                # where the default template file won't be found alongside it.
                return str(Path("/tmp/some_other_random_nonexistent_place") / "dummy_template_manager.py")
            return original_os_path_abspath(path_arg)

        monkeypatch.setattr(llm_service.prompts.template_manager.os.path, "abspath", mock_abspath_for_nonexistent_default)

        # This should not raise an exception (because templates_file_path is None)
        # but log an error and start with empty templates.
        manager = PromptTemplateManager(templates_file_path=None)
        assert manager.templates == {}
        assert manager.list_template_names() == []
        # The _templates_file_path would be set to the default non-existent path
        # This path comes from the mock_abspath_for_nonexistent_default function
        expected_default_path = Path("/tmp/some_other_random_nonexistent_place") / "prompt_templates.yaml"
        assert manager._templates_file_path == str(expected_default_path)
