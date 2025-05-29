# This file makes mcp_service_manager a Python package.

from .exceptions import (
    MCPError,
    MCPConfigError,
    MCPFileNotFoundError,
    MCPJSONDecodeError,
    MCPServiceRegistryError,
    MCPPlatformUnavailableError,
    MCPInstallationError,
    MCPDownloadError,
    MCPExtractionError,
    MCPAlreadyInstalledError,
    MCPNotFoundError,
    MCPLoadError,
    MCPManifestError,
    MCPModuleNotFoundError,
    MCPClassNotFoundError,
    MCPPluginInitializationError
)

from .core import MCP, MCPConfig, AbstractMCPPlugin
from .registry import ServiceRegistry
from .installer import MCPInstaller
from .monitor import ServiceMonitor
from .loader import PluginLoader
from .logging_setup import setup_logging
