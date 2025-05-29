import json
import importlib
import sys
from pathlib import Path
from typing import Optional # Already imported in AbstractMCPPlugin for MCP type hint
import logging

from .core import AbstractMCPPlugin, MCPConfig
from .models import InstalledMCP # SQLAlchemy model
from .exceptions import (
    MCPLoadError,
    MCPManifestError,
    MCPModuleNotFoundError,
    MCPClassNotFoundError,
    MCPPluginInitializationError,
    MCPFileNotFoundError, # For MCPConfig
    MCPJSONDecodeError    # For MCPConfig
)

logger = logging.getLogger("mcp_manager")

class PluginLoader:
    """
    Responsible for dynamically loading and instantiating MCP plugins
    based on a manifest file within the installed MCP package.
    """
    MANIFEST_FILE_NAME = "mcp_manifest.json"

    def load_mcp_plugin(self, installed_mcp: InstalledMCP) -> AbstractMCPPlugin:
        """
        Loads and instantiates the main plugin class for a given installed MCP.

        The MCP package is expected to contain an 'mcp_manifest.json' file
        at the root of its local_path, specifying the entry point module and class.

        Args:
            installed_mcp: An InstalledMCP SQLAlchemy object containing metadata
                           about the installed MCP, including its local_path and
                           config_file_path.

        Returns:
            An instance of the MCP's plugin class (which implements AbstractMCPPlugin).

        Raises:
            MCPManifestError: If the manifest file is missing, malformed, or does not
                              contain the required 'entry_point' information.
            MCPModuleNotFoundError: If the module specified in the manifest cannot be imported.
            MCPClassNotFoundError: If the class specified in the manifest cannot be found
                                   in the imported module.
            MCPPluginInitializationError: If the plugin class cannot be instantiated,
                                          or if MCPConfig fails to load.
            MCPLoadError: For other general loading issues.
        """
        logger.info(f"Attempting to load plugin for MCP: {installed_mcp.mcp_name} from path: {installed_mcp.local_path}")
        mcp_local_path = Path(installed_mcp.local_path)
        manifest_path = mcp_local_path / self.MANIFEST_FILE_NAME

        if not manifest_path.is_file():
            logger.error(f"Manifest file '{self.MANIFEST_FILE_NAME}' not found at {manifest_path} for MCP {installed_mcp.mcp_name}.")
            raise MCPManifestError(
                path=str(manifest_path),
                details=f"Manifest file '{self.MANIFEST_FILE_NAME}' not found."
            )

        logger.debug(f"Reading manifest file: {manifest_path}")
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            logger.debug(f"Manifest file read successfully for {installed_mcp.mcp_name}.")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in manifest file {manifest_path} for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPManifestError(
                path=str(manifest_path),
                details=f"Invalid JSON in manifest file: {e}"
            )
        except IOError as e:
            logger.error(f"Could not read manifest file {manifest_path} for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPManifestError(
                path=str(manifest_path),
                details=f"Could not read manifest file: {e}"
            )

        entry_point = manifest_data.get("entry_point")
        if not isinstance(entry_point, dict) or \
           "module" not in entry_point or "class" not in entry_point:
            logger.error(f"Invalid 'entry_point' in manifest {manifest_path} for MCP {installed_mcp.mcp_name}.")
            raise MCPManifestError(
                path=str(manifest_path),
                details="Manifest file must contain an 'entry_point' object "
                        "with 'module' and 'class' keys."
            )

        module_name = entry_point["module"]
        class_name = entry_point["class"]
        logger.debug(f"Manifest entry point for {installed_mcp.mcp_name}: module='{module_name}', class='{class_name}'")

        # Temporarily add the MCP's local path to sys.path to allow import
        # Ensure it's a string, as sys.path expects strings.
        str_mcp_local_path = str(mcp_local_path.resolve())

        # Load MCPConfig first, as it's needed for plugin initialization
        logger.debug(f"Loading MCPConfig for plugin {class_name} of MCP {installed_mcp.mcp_name} using config: {installed_mcp.config_file_path}")
        try:
            mcp_config_loader = MCPConfig(config_file_path=installed_mcp.config_file_path)
            mcp_config_loader.load_config() # This can raise MCPFileNotFoundError, MCPJSONDecodeError
        except (MCPFileNotFoundError, MCPJSONDecodeError) as e:
            logger.error(f"Failed to load MCPConfig for plugin {class_name} of MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPPluginInitializationError(
                class_name=class_name,
                details=f"Failed to load MCPConfig for plugin: {e}"
            )
        except Exception as e: # Catch any other unexpected error during config load
            logger.error(f"Unexpected error loading MCPConfig for plugin {class_name}, MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPPluginInitializationError(
                class_name=class_name,
                details=f"Unexpected error loading MCPConfig: {e}"
            )

        path_added_to_sys = False
        if str_mcp_local_path not in sys.path:
            sys.path.insert(0, str_mcp_local_path)
            path_added_to_sys = True
            logger.debug(f"Added {str_mcp_local_path} to sys.path for importing {module_name}")
        
        plugin_module = None
        logger.debug(f"Importing module '{module_name}' for MCP {installed_mcp.mcp_name}")
        try:
            plugin_module = importlib.import_module(module_name)
            logger.debug(f"Module '{module_name}' imported successfully.")
        except ImportError as e:
            logger.error(f"Failed to import module '{module_name}' from {str_mcp_local_path} for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPModuleNotFoundError(
                module_name=module_name,
                path=str_mcp_local_path,
                details=f"Failed to import module: {e}"
            )
        except Exception as e: # Catch any other unexpected error during import
            logger.error(f"Unexpected error importing module '{module_name}' from {str_mcp_local_path} for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPModuleNotFoundError(
                module_name=module_name,
                path=str_mcp_local_path,
                details=f"An unexpected error occurred during module import: {e}"
            )
        finally:
            if path_added_to_sys:
                # Clean up sys.path
                if str_mcp_local_path in sys.path:
                    sys.path.remove(str_mcp_local_path)
                    logger.debug(f"Removed {str_mcp_local_path} from sys.path")

        if plugin_module is None: # Should be caught by exceptions above, but as a safeguard
            # This line is technically unreachable if the above try/except/finally works as expected.
            logger.critical(f"Module '{module_name}' for MCP {installed_mcp.mcp_name} is None after import attempt without specific error.")
            raise MCPLoadError(f"Module '{module_name}' could not be loaded for unknown reasons.")

        logger.debug(f"Getting class '{class_name}' from module '{module_name}' for MCP {installed_mcp.mcp_name}")
        try:
            plugin_class = getattr(plugin_module, class_name)
            logger.debug(f"Class '{class_name}' retrieved successfully.")
        except AttributeError as e:
            logger.error(f"Class '{class_name}' not found in module '{module_name}' for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPClassNotFoundError(
                class_name=class_name,
                module_name=module_name,
                details=f"Class '{class_name}' not found in module '{module_name}'."
            )
        except Exception as e: # Catch any other unexpected error during getattr
            logger.error(f"Unexpected error getting class '{class_name}' from module '{module_name}' for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPClassNotFoundError(
                class_name=class_name,
                module_name=module_name,
                details=f"An unexpected error occurred while getting class attribute: {e}"
            )

        logger.debug(f"Instantiating plugin class '{class_name}' for MCP {installed_mcp.mcp_name}")
        try:
            plugin_instance = plugin_class(mcp_config=mcp_config_loader)
            logger.info(f"Plugin '{class_name}' for MCP {installed_mcp.mcp_name} instantiated successfully.")
        except Exception as e: # Catch errors during plugin instantiation
            logger.error(f"Error during instantiation of plugin '{class_name}' for MCP {installed_mcp.mcp_name}: {e}", exc_info=True)
            raise MCPPluginInitializationError(
                class_name=class_name,
                details=f"Error during instantiation of plugin '{class_name}': {e}"
            )

        if not isinstance(plugin_instance, AbstractMCPPlugin):
            logger.error(f"Loaded plugin '{class_name}' for MCP {installed_mcp.mcp_name} does not implement AbstractMCPPlugin.")
            raise MCPLoadError(
                f"Loaded plugin '{class_name}' from '{module_name}' does not implement AbstractMCPPlugin."
            )
        
        logger.info(f"Successfully loaded plugin for MCP: {installed_mcp.mcp_name} (Class: {class_name})")
        return plugin_instance
