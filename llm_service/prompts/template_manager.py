import yaml
import os
import logging
from typing import Dict, Optional, Any

# Adjusting import path based on the new directory structure
# Assuming exceptions are in llm_service.exceptions
from ..exceptions import PromptTemplateError, PromptTemplateNotFoundError

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATES_FILENAME = "prompt_templates.yaml"

class PromptTemplateManager:
    """
    Manages loading and formatting of prompt templates from a YAML file.
    """
    def __init__(self, templates_file_path: Optional[str] = None):
        """
        Initializes the PromptTemplateManager.

        Args:
            templates_file_path: Path to the YAML file containing templates.
                                 If None, defaults to 'prompt_templates.yaml'
                                 in the same directory as this manager.
        """
        self.templates: Dict[str, str] = {}

        if templates_file_path is None:
            # Default to `prompt_templates.yaml` in the same directory as this file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            self._templates_file_path = os.path.join(base_dir, DEFAULT_TEMPLATES_FILENAME)
        else:
            self._templates_file_path = templates_file_path

        try:
            self.load_templates_from_file(self._templates_file_path)
        except PromptTemplateError as e: # Catch specific error from load_templates
             logger.error(f"Initial template loading failed: {e}")
             # Depending on desired behavior, could re-raise or start with empty templates
             # For now, it will start with empty templates if default file is problematic
             # and allow explicit load_templates_from_file later.
             # If a specific path was provided and fails, it should probably raise.
             if templates_file_path is not None: # Only raise if a custom path was given
                 raise

    def load_templates_from_file(self, file_path: str) -> None:
        """
        Loads prompt templates from the specified YAML file.

        Args:
            file_path: The path to the YAML template file.

        Raises:
            PromptTemplateError: If the file is not found, or if there's an error
                                 parsing the YAML content.
        """
        logger.info(f"Loading prompt templates from: {file_path}")
        self.templates = {} # Clear existing templates before loading new ones
        try:
            with open(file_path, 'r') as f:
                loaded_data = yaml.safe_load(f)
                if loaded_data is None: # Handle empty YAML file
                    logger.info(f"Templates file '{file_path}' is empty. No templates loaded.")
                    # self.templates is already cleared, so it remains empty.
                    # self._templates_file_path is updated below.
                elif not isinstance(loaded_data, dict):
                    raise PromptTemplateError(f"Templates file '{file_path}' content is not a valid YAML dictionary (mapping). Found type: {type(loaded_data)}")
                else:
                    # Ensure all template values are strings
                    for key, value in loaded_data.items():
                        if not isinstance(value, str):
                            logger.warning(f"Template '{key}' in '{file_path}' is not a string (type: {type(value)}). It will be converted or skipped.")
                            # Option: convert, skip, or raise error. For now, try converting.
                            try:
                                self.templates[str(key)] = str(value)
                            except Exception:
                                 raise PromptTemplateError(f"Template '{key}' in '{file_path}' could not be coerced to string.")
                        else:
                            self.templates[str(key)] = value

            logger.info(f"Successfully loaded {len(self.templates)} templates from {file_path}.")
            self._templates_file_path = file_path # Update path if successfully loaded
        except FileNotFoundError:
            logger.error(f"Prompt templates file not found: {file_path}")
            raise PromptTemplateError(f"Prompt templates file not found: {file_path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML from prompt templates file '{file_path}': {e}", exc_info=True)
            raise PromptTemplateError(f"Error parsing YAML from prompt templates file '{file_path}': {e}")
        except Exception as e: # Catch any other unexpected errors during loading
            logger.error(f"An unexpected error occurred loading templates from '{file_path}': {e}", exc_info=True)
            raise PromptTemplateError(f"An unexpected error occurred loading templates from '{file_path}': {e}")


    def get_template(self, template_name: str) -> str:
        """
        Retrieves the raw template string for the given template name.

        Args:
            template_name: The name of the template to retrieve.

        Returns:
            The raw template string.

        Raises:
            PromptTemplateNotFoundError: If the template name is not found.
        """
        if template_name not in self.templates:
            logger.warning(f"Prompt template '{template_name}' not found.")
            raise PromptTemplateNotFoundError(template_name)
        return self.templates[template_name]

    def format_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Formats a prompt template with the given keyword arguments.

        Args:
            template_name: The name of the template to format.
            **kwargs: Keyword arguments representing the values for placeholders
                      in the template.

        Returns:
            The formatted prompt string.

        Raises:
            PromptTemplateNotFoundError: If the template name is not found.
            PromptTemplateError: If a required placeholder is missing in kwargs (KeyError).
        """
        template_string = self.get_template(template_name) # Handles NotFoundError
        try:
            formatted_string = template_string.format(**kwargs)
            return formatted_string
        except KeyError as e:
            logger.error(f"Missing placeholder value for template '{template_name}': {e}", exc_info=True)
            # Re-raise as a more specific PromptTemplateError
            raise PromptTemplateError(f"Missing value for placeholder {e} in template '{template_name}'. Required: {kwargs.keys()}")
        except Exception as e: # Catch any other formatting errors
            logger.error(f"An unexpected error occurred formatting template '{template_name}': {e}", exc_info=True)
            raise PromptTemplateError(f"An unexpected error formatting template '{template_name}': {e}")

    def list_template_names(self) -> list[str]:
        """Returns a list of available template names."""
        return list(self.templates.keys())

# Example usage (conceptual)
# if __name__ == '__main__':
#     logging.basicConfig(level=logging.INFO)
#     # Assuming prompt_templates.yaml is in the same directory
#     try:
#         manager = PromptTemplateManager()
#         print("Loaded templates:", manager.list_template_names())

#         # Test sql_generation
#         sql_template_name = "sql_generation"
#         sql_params = {"schema": "CREATE TABLE users (id INT, name VARCHAR(50))", "question": "Show me all users"}
#         formatted_sql = manager.format_template(sql_template_name, **sql_params)
#         print(f"\nFormatted '{sql_template_name}':\n{formatted_sql}")

#         # Test summarization
#         summary_template_name = "summarization"
#         summary_params = {"text": "This is a long piece of text that needs to be summarized effectively."}
#         formatted_summary = manager.format_template(summary_template_name, **summary_params)
#         print(f"\nFormatted '{summary_template_name}':\n{formatted_summary}")

#         # Test missing placeholder
#         try:
#             manager.format_template(sql_template_name, schema="CREATE TABLE products (id INT, price DECIMAL)")
#         except PromptTemplateError as e:
#             print(f"\nError (missing placeholder 'question'): {e}")

#         # Test non-existent template
#         try:
#             manager.get_template("non_existent_template")
#         except PromptTemplateNotFoundError as e:
#             print(f"\nError (template not found): {e}")

#     except PromptTemplateError as e:
#         print(f"Failed to initialize or use PromptTemplateManager: {e}")

#     # Test with a specific file path (if you want to place the YAML elsewhere)
#     # specific_path = "path/to/your/templates.yaml"
#     # try:
#     # manager_specific = PromptTemplateManager(templates_file_path=specific_path)
#     # ...
#     # except PromptTemplateError as e:
#     # print(f"Failed with specific path: {e}")
