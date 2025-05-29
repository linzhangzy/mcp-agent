# MCP Service Manager - Developer Guide

## 1. Introduction

The `mcp_service_manager` module is a core component designed to manage the lifecycle of Modular Control Programs (MCPs) within a larger AI Agent system. Its primary purpose is to provide functionalities for discovering, installing, configuring, enabling/disabling, monitoring, and uninstalling various MCPs. These MCPs are envisioned as plugins that extend the agent's capabilities.

This guide provides an overview of the module's architecture, its components, API usage, and detailed instructions for developing new MCP plugins compatible with this system.

## 2. Architecture Overview

The `mcp_service_manager` operates on a modular architecture:

*   **API Layer (`api_v1.py`):** Exposes FastAPI endpoints for external management of MCPs.
*   **Core Services:**
    *   `ServiceRegistry`: Handles discovery of available MCPs from a central repository (e.g., `mcp.so`).
    *   `MCPInstaller`: Manages the local installation, uninstallation, and state (enabled/disabled) of MCPs. This includes downloading MCP packages, extracting them, and managing their file structure.
    *   `PluginLoader`: Dynamically loads the code for installed MCPs based on a manifest file.
    *   `ServiceMonitor`: Provides status information and details about installed MCPs.
*   **MCP Plugin Interface (`AbstractMCPPlugin` in `core.py`):** Defines the contract that all MCP plugins must adhere to.
*   **Configuration Management (`MCPConfig` in `core.py`):** Handles loading and accessing instance-specific JSON configuration for each MCP.
*   **Data Persistence (`models.py`):** Uses SQLAlchemy to store metadata about installed MCPs in a database (via the `InstalledMCP` model).

**Key Directories & Files:**

*   `mcp_service_manager/`: Root directory for the module.
    *   `api_v1.py`: FastAPI router and endpoint implementations.
    *   `core.py`: Contains `MCP`, `MCPConfig`, and `AbstractMCPPlugin`.
    *   `installer.py`: `MCPInstaller` class.
    *   `loader.py`: `PluginLoader` class.
    *   `registry.py`: `ServiceRegistry` class.
    *   `monitor.py`: `ServiceMonitor` class.
    *   `models.py`: SQLAlchemy models (e.g., `InstalledMCP`).
    *   `exceptions.py`: Custom exceptions for the module.
    *   `schemas.py`: Pydantic schemas for API request/response validation.
    *   `logging_setup.py`: Configures logging for the module.
    *   `DEVELOPER_GUIDE.md`: This document.
*   `logs/`: Directory (created at runtime in the current working directory) containing `app.log` and `mcp_operations.log`.
*   `/app/installed_mcps/` (default): Default base directory where MCP packages are installed. Each MCP gets a subdirectory named after it, containing its versioned content and configuration.

## 3. Core Components

*   **`ServiceRegistry` (`registry.py`):**
    *   Responsible for fetching a list of available MCPs from a central platform (defaulting to `https://mcp.so/api/servers`).
    *   Implements a simple time-based caching mechanism to reduce frequent external requests.
*   **`MCPConfig` (`core.py`):**
    *   Manages the loading, parsing, and accessing of an MCP's instance-specific JSON configuration file (typically `mcp_config.json` within the MCP's installation directory).
    *   Provides `load_config()`, `get()`, and `get_all()` methods.
    *   Raises custom exceptions for file not found or JSON decoding errors.
*   **`MCPInstaller` (`installer.py`):**
    *   Handles the download of MCP packages (zip or tar.gz) from a given URL.
    *   Extracts the package contents into a structured directory: `<base_install_path>/<mcp_name>/<version>/mcp_content/`.
    *   Creates an `mcp_config.json` file for the MCP instance: `<base_install_path>/<mcp_name>/mcp_config.json`.
    *   Records MCP metadata in the database using the `InstalledMCP` model.
    *   Manages enabling/disabling of MCPs by updating the `is_enabled` flag in the database.
    *   Handles uninstallation by removing the MCP's files and its database record.
*   **`PluginLoader` (`loader.py`):**
    *   Dynamically loads an MCP's plugin code at runtime.
    *   Reads an `mcp_manifest.json` file from the root of the MCP's installed content directory.
    *   The manifest specifies the entry point module and class for the plugin.
    *   Adds the MCP's content directory to `sys.path` temporarily for import.
    *   Instantiates the plugin class, passing an `MCPConfig` object to its constructor.
*   **`AbstractMCPPlugin` (`core.py`):**
    *   An Abstract Base Class (ABC) defining the interface that all MCP plugins must implement.
    *   Ensures that all plugins provide a consistent set of methods for lifecycle management and status reporting.
    *   Key methods: `__init__(self, mcp_config)`, `get_status()`, `on_enable()`, `on_disable()`.
*   **`ServiceMonitor` (`monitor.py`):**
    *   Provides methods to retrieve status and details of installed MCPs from the database.
    *   Does not directly interact with plugin code but relies on the `InstalledMCP` records. (Future enhancement: could call `plugin.get_status()` if a plugin is loaded).
*   **Database Models (`models.py` - `InstalledMCP`):**
    *   SQLAlchemy model representing an installed MCP instance.
    *   Stores metadata such as name, version, installation path, configuration file path, and enabled status.

## 4. API Endpoints

The `mcp_service_manager.api_v1` module provides RESTful API endpoints for managing MCPs. Key endpoints include:

*   `GET /api/v1/mcp/available`: Lists MCPs available for installation from the central registry.
*   `GET /api/v1/mcp/installed`: Lists all currently installed MCPs.
*   `POST /api/v1/mcp/install`: Installs a new MCP package. Requires details like MCP name, version, download URL, etc.
*   `GET /api/v1/mcp/{mcp_name}`: Retrieves status and details for a specific installed MCP.
*   `PUT /api/v1/mcp/{mcp_name}/config`: Updates the `mcp_config.json` file for a given MCP.
*   `POST /api/v1/mcp/{mcp_name}/enable`: Enables a disabled MCP.
*   `POST /api/v1/mcp/{mcp_name}/disable`: Disables an enabled MCP.
*   `DELETE /api/v1/mcp/{mcp_name}`: Uninstalls an MCP.

For detailed request/response schemas and trying out the API, refer to the OpenAPI/Swagger documentation typically available at `/docs` when the FastAPI application is running.

## 5. Logging

The module uses Python's standard `logging` library.
*   The `setup_logging()` function in `mcp_service_manager.logging_setup` configures the loggers. It's intended to be called once at application startup.
*   Two primary loggers are configured:
    *   `agent_core`: For general application logs (writes to `logs/app.log`).
    *   `mcp_manager`: For logs specific to MCP service management operations (writes to `logs/mcp_operations.log` and also to `logs/app.log`).
*   Log files are created in a `logs/` subdirectory in the current working directory of the application.
*   Log handlers use `RotatingFileHandler` with a default max size of 10MB and 5 backup files.

## 6. Testing

The module includes a suite of tests:
*   **Unit Tests (`tests/unit`):** Focus on individual components in isolation (e.g., `MCPConfig`, `ServiceRegistry`, `MCPInstaller` helpers). These use mocking extensively.
*   **Integration Tests (`tests/integration`):** Test the interaction between components, particularly the API endpoints and their connection to the service layer and database. These use an in-memory SQLite database and a FastAPI `TestClient`.

Developers adding new features or modifying existing ones should include relevant unit and/or integration tests.

## 7. Developing a New MCP Plugin

This is the most crucial section for developers wishing to extend the system with new MCP capabilities. To create a new MCP compatible with the `mcp_service_manager`, follow these steps:

### 7.1. Implement `AbstractMCPPlugin`

Your MCP plugin **must** provide a class that inherits from `mcp_service_manager.core.AbstractMCPPlugin` and implements all its abstract methods.

```python
# Example: your_mcp_package/plugin_impl.py
import logging
from mcp_service_manager.core import AbstractMCPPlugin, MCPConfig

# It's good practice for plugins to use their own logger or the mcp_manager logger
logger = logging.getLogger("mcp_manager") # Or a more specific plugin logger

class YourMCPPluginClass(AbstractMCPPlugin):
    def __init__(self, mcp_config: MCPConfig):
        """
        Constructor for your plugin.
        The mcp_config object is pre-loaded with the content of this MCP's 
        mcp_config.json file.
        """
        super().__init__(mcp_config)
        self.name = self.mcp_config.get("mcp_instance_name", "DefaultPluginName")
        # Load other configurations specific to your plugin
        self.api_key = self.mcp_config.get("api_key")
        self.target_url = self.mcp_config.get("target_url", "https://api.example.com/default")
        
        logger.info(f"Plugin {self.name} initialized with target URL: {self.target_url}")
        # Perform any other setup that relies on the initial config but doesn't
        # require active connections (which should go into on_enable).

    def get_status(self) -> dict:
        """
        Return the operational status of your plugin.
        This could include health checks, connection status, loaded resources, etc.
        Example:
        return {
            "name": self.name,
            "connection_status": "connected" if self.is_connected else "disconnected",
            "custom_metric": self.custom_metric_value
        }
        """
        logger.debug(f"get_status called for {self.name}")
        # Implement actual status checking logic here
        return {
            "name": self.name,
            "status": "operational", # Replace with actual status logic
            "config_target_url": self.target_url,
            "api_key_present": bool(self.api_key)
        }

    def on_enable(self):
        """
        Called when the MCP is enabled.
        Use this to perform setup tasks, establish connections, start threads, etc.
        """
        logger.info(f"Plugin {self.name} is being enabled.")
        # Example: Initialize connections, load resources
        # self.connect_to_service()
        # self.load_data_models()
        pass # Replace with actual enable logic

    def on_disable(self):
        """
        Called when the MCP is disabled.
        Use this to perform cleanup tasks, release resources, stop threads, etc.
        """
        logger.info(f"Plugin {self.name} is being disabled.")
        # Example: Close connections, release resources
        # self.disconnect_from_service()
        # self.unload_data_models()
        pass # Replace with actual disable logic

    # You can add other methods specific to your plugin's functionality
    # def do_specific_task(self, data: Any) -> Any:
    #     logger.info(f"Plugin {self.name} performing specific task with data: {data}")
    #     # ... your plugin's core logic ...
    #     return processed_data
```

### 7.2. Create `mcp_manifest.json`

At the root of your MCP package's directory structure (which will also be the root of your distributable archive), you **must** include a manifest file named `mcp_manifest.json`. This file tells the `PluginLoader` how to load your plugin.

**Structure:**

```json
{
  "mcp_name": "UniqueNameOfYourMCP",
  "version": "1.0.0",
  "display_name": "My Awesome MCP",
  "description": "A brief description of what your MCP does.",
  "entry_point": {
    "module": "your_mcp_package.plugin_impl",
    "class": "YourMCPPluginClass"
  },
  "author": "Your Name/Organization",
  "license": "Apache-2.0"
}
```

*   **`mcp_name` (string, required):** A unique identifier for your MCP. This should be distinct from other MCPs.
*   **`version` (string, required):** The version of your MCP (e.g., semantic versioning like "1.0.0").
*   **`display_name` (string, optional):** A human-readable name for display purposes.
*   **`description` (string, optional):** A short description of the MCP's functionality.
*   **`entry_point` (object, required):** This object is critical for the `PluginLoader`.
    *   **`module` (string, required):** The Python module path to your main plugin file, relative to the root of your MCP package. For example, if your plugin class `YourMCPPluginClass` is in `your_mcp_package/plugin_impl.py`, this would be `"your_mcp_package.plugin_impl"`. If it's in `main.py` at the root, it would be `"main"`.
    *   **`class` (string, required):** The name of the class within the specified module that implements `AbstractMCPPlugin`.
*   **`author` (string, optional):** The author or organization responsible for the MCP.
*   **`license` (string, optional):** The license under which the MCP is distributed.

### 7.3. Package the MCP

Your MCP plugin, including all its code, dependencies (if bundled), and the `mcp_manifest.json` file, should be packaged into a single archive file.
*   **Supported Formats:** `.zip` or `.tar.gz`
*   **Structure:** The `mcp_manifest.json` file **must** be at the root level of the archive. All Python package code should also be relative to this root.

**Example Archive Structure (`my_awesome_mcp.zip`):**
```
my_awesome_mcp.zip
├── mcp_manifest.json
└── your_mcp_package/
    ├── __init__.py
    ├── plugin_impl.py  (contains YourMCPPluginClass)
    ├── utils.py
    └── ... (other modules or resources)
```

### 7.4. Configuration (`mcp_config.json`)

When an MCP is installed via the `MCPInstaller` (e.g., through the API), an `mcp_config.json` file is created for that specific instance. The path to this file is ` <base_install_path>/<mcp_name>/mcp_config.json`.

*   The `MCPInstaller` can optionally populate this file from a `config_template` provided during the install request.
*   Your plugin's `__init__` method receives an `MCPConfig` object that is already initialized with the path to this instance-specific configuration file. Your plugin should call `self.mcp_config.load_config()` to load the data.
*   Your plugin can then use `self.mcp_config.get("your_key", default_value)` to retrieve its settings.
*   This configuration file can be updated at runtime via the `PUT /api/v1/mcp/{mcp_name}/config` API endpoint. Your plugin should be prepared to handle configuration changes, potentially by re-reading relevant parts of the config when methods like `on_enable` are called or by implementing a dedicated reconfigure method if needed.

## 8. Future Considerations

*   **Plugin-Specific Dependencies:** The current model assumes plugins either bundle their dependencies or rely on system-wide installed packages. A more robust dependency management system per plugin could be considered.
*   **Granular Plugin Capabilities:** The manifest could be extended to declare specific capabilities or permissions the plugin requires.
*   **Hot-Reloading/Updating:** Currently, updating an MCP likely requires an uninstall and reinstall. More sophisticated update mechanisms could be explored.
*   **Inter-Plugin Communication:** No formal mechanism for direct communication between MCP plugins is defined.
*   **Security:** Dynamic loading of code requires careful consideration of security implications. MCPs should be sourced from trusted locations.

---
This guide should provide developers with the necessary information to understand and extend the MCP Service Manager.
