class MCPError(Exception):
    """Base exception for all mcp_service_manager errors."""
    pass

class MCPConfigError(MCPError):
    """Base exception for MCP configuration errors."""
    pass

class MCPFileNotFoundError(MCPConfigError):
    """Raised when the MCP configuration file is not found."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        super().__init__(f"Configuration file not found: {filepath}")

class MCPJSONDecodeError(MCPConfigError):
    """Raised when the MCP configuration file is not valid JSON."""
    def __init__(self, filepath: str, original_exception: Exception):
        self.filepath = filepath
        self.original_exception = original_exception
        super().__init__(f"Error decoding JSON from configuration file: {filepath}. Details: {original_exception}")

class MCPInstallationError(MCPError):
    """Base exception for MCP installation errors."""
    pass

class MCPDownloadError(MCPInstallationError):
    """Raised when downloading an MCP package fails."""
    def __init__(self, url: str, details: str):
        self.url = url
        self.details = details
        super().__init__(f"Failed to download MCP from {url}. Details: {details}")

class MCPExtractionError(MCPInstallationError):
    """Raised when extracting an MCP package fails."""
    def __init__(self, package_path: str, details: str):
        self.package_path = package_path
        self.details = details
        super().__init__(f"Failed to extract MCP package {package_path}. Details: {details}")

class MCPAlreadyInstalledError(MCPInstallationError):
    """Raised when attempting to install an MCP that is already installed."""
    def __init__(self, mcp_name: str):
        self.mcp_name = mcp_name
        super().__init__(f"MCP '{mcp_name}' is already installed.")

class MCPNotFoundError(MCPInstallationError): # Or MCPError if preferred as a more general "not found"
    """Raised when an MCP is not found in the database or filesystem."""
    def __init__(self, mcp_name: str = None, mcp_id: int = None, details: str = None):
        if mcp_name:
            message = f"MCP '{mcp_name}' not found."
        elif mcp_id:
            message = f"MCP with ID '{mcp_id}' not found."
        elif details:
            message = details
        else:
            message = "MCP not found."
        super().__init__(message)

class MCPLoadError(MCPError):
    """Base exception for errors encountered while loading an MCP plugin."""
    pass

class MCPManifestError(MCPLoadError):
    """Raised when there's an issue with the MCP's manifest file."""
    def __init__(self, path: str, details: str):
        self.path = path
        self.details = details
        super().__init__(f"Error with manifest file at {path}: {details}")

class MCPModuleNotFoundError(MCPLoadError):
    """Raised when the specified plugin module cannot be found or imported."""
    def __init__(self, module_name: str, path: str, details: str):
        self.module_name = module_name
        self.path = path
        self.details = details
        super().__init__(f"Cannot find or import module '{module_name}' from path '{path}'. Details: {details}")

class MCPClassNotFoundError(MCPLoadError):
    """Raised when the specified plugin class cannot be found in the module."""
    def __init__(self, class_name: str, module_name: str, details: str):
        self.class_name = class_name
        self.module_name = module_name
        self.details = details
        super().__init__(f"Cannot find class '{class_name}' in module '{module_name}'. Details: {details}")

class MCPPluginInitializationError(MCPLoadError):
    """Raised when a plugin fails to initialize."""
    def __init__(self, class_name: str, details: str):
        self.class_name = class_name
        self.details = details
        super().__init__(f"Failed to initialize plugin class '{class_name}'. Details: {details}")

class MCPServiceRegistryError(MCPError):
    """Base exception for MCP service registry errors."""
    pass

class MCPPlatformUnavailableError(MCPServiceRegistryError):
    """Raised when the MCP platform (e.g., mcp.so) is unavailable or returns an error."""
    def __init__(self, url: str, details: str):
        self.url = url
        self.details = details
        super().__init__(f"MCP platform at {url} is unavailable. Details: {details}")
