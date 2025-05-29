import json
import abc # For Abstract Base Class
from typing import Optional, Dict, Any # For type hinting
import logging

from .exceptions import MCPFileNotFoundError, MCPJSONDecodeError

logger = logging.getLogger("mcp_manager")

# Forward declaration for type hint in AbstractMCPPlugin
class MCPConfig:
    pass

class AbstractMCPPlugin(abc.ABC):
    """
    Abstract Base Class for MCP plugins.
    Defines the interface that all concrete MCP plugins must implement.
    """
    def __init__(self, mcp_config: MCPConfig):
        """
        Constructor for the plugin.

        Args:
            mcp_config: An MCPConfig instance containing the specific configuration
                        for this MCP plugin.
        """
        self.mcp_config = mcp_config

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """
        Returns the operational status of the plugin.
        This can include health checks, connection status, etc.

        Returns:
            A dictionary representing the status.
        """
        pass

    @abc.abstractmethod
    def on_enable(self):
        """
        Called when the MCP is enabled.
        Plugins can use this to perform setup tasks, establish connections, etc.
        """
        pass

    @abc.abstractmethod
    def on_disable(self):
        """
        Called when the MCP is disabled.
        Plugins can use this to perform cleanup tasks, release resources, etc.
        """
        pass

class MCP:
    """
    Represents an MCP service instance, potentially with a loaded plugin.
    This class holds metadata about an MCP and can hold a reference to its
    active plugin if loaded.
    """
    def __init__(self, name: str, version: str, source_url: str, local_path: str,
                 config_file_path: str, plugin: Optional[AbstractMCPPlugin] = None):
        self.name = name
        self.version = version
        self.source_url = source_url
        self.local_path = local_path
        self.config_file_path = config_file_path
        self.plugin: Optional[AbstractMCPPlugin] = plugin

class MCPConfig:
    """Handles the loading, parsing, and accessing of an MCP's JSON configuration file."""
    def __init__(self, config_file_path: str):
        """
        Initializes the MCPConfig with the path to the configuration file.

        Args:
            config_file_path: The path to the MCP's JSON configuration file.
        """
        self.config_file_path = config_file_path
        self.config_data = None
        logger.debug(f"MCPConfig initialized for file: {config_file_path}")

    def load_config(self):
        """
        Reads and parses the JSON configuration file.

        Stores the parsed configuration in self.config_data.

        Raises:
            MCPFileNotFoundError: If the configuration file is not found.
            MCPJSONDecodeError: If the configuration file is not valid JSON.
        """
        logger.info(f"Loading MCP configuration from: {self.config_file_path}")
        try:
            with open(self.config_file_path, 'r') as f:
                self.config_data = json.load(f)
            logger.info(f"Successfully loaded configuration from: {self.config_file_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_file_path}", exc_info=True)
            raise MCPFileNotFoundError(self.config_file_path)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {self.config_file_path}: {e}", exc_info=True)
            raise MCPJSONDecodeError(self.config_file_path, e)
        # Optional: Basic validation placeholder
        # if self.config_data and "expected_key" not in self.config_data:
        #     logger.warning(f"'expected_key' not found in {self.config_file_path}")

    def get(self, key: str, default=None):
        """
        Retrieves a specific configuration value by its key.

        Args:
            key: The key of the configuration value to retrieve.
            default: The default value to return if the key is not found.

        Returns:
            The configuration value if the key is found, otherwise the default value.
            Returns default if config_data has not been loaded yet.
        """
        if self.config_data is None:
            return default
        return self.config_data.get(key, default)

    def get_all(self) -> dict:
        """
        Returns the entire loaded configuration dictionary.

        Returns:
            A dictionary containing all configuration data, or an empty dictionary
            if the configuration has not been loaded or is empty.
        """
        return self.config_data if self.config_data is not None else {}
