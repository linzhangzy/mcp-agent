import requests
import time
import json # For json.JSONDecodeError
import logging

from .exceptions import MCPServiceRegistryError, MCPPlatformUnavailableError

logger = logging.getLogger("mcp_manager")

class ServiceRegistry:
    """
    Handles discovery of available MCP services from a central platform (e.g., mcp.so)
    and includes a simple time-based caching mechanism.
    """
    DEFAULT_MCP_SO_URL = "https://mcp.so/"
    DEFAULT_API_ENDPOINT = "api/servers"
    DEFAULT_CACHE_DURATION_SECONDS = 3600  # 1 hour

    def __init__(self,
                 mcp_so_url: str = DEFAULT_MCP_SO_URL,
                 cache_duration_seconds: int = DEFAULT_CACHE_DURATION_SECONDS):
        """
        Initializes the ServiceRegistry.

        Args:
            mcp_so_url: The base URL for the MCP discovery platform.
            cache_duration_seconds: Duration in seconds to cache the service list.
        """
        if not mcp_so_url.endswith('/'):
            mcp_so_url += '/'
        self.mcp_so_url = mcp_so_url
        self.api_endpoint = self.DEFAULT_API_ENDPOINT
        self.cache_duration_seconds = cache_duration_seconds

        self._cached_services: list | None = None
        self._last_fetch_time: float = 0.0
        logger.debug(f"ServiceRegistry initialized with URL: {self.mcp_so_url}, cache duration: {cache_duration_seconds}s")

    def list_available_services(self) -> list:
        """
        Fetches a list of available MCP services from the configured platform.
        Uses a time-based cache to avoid frequent requests.

        Returns:
            A list of dictionaries, where each dictionary represents an MCP service.

        Raises:
            MCPPlatformUnavailableError: If the platform is unavailable or returns an error.
            MCPServiceRegistryError: If there's an issue processing the platform's response (e.g., invalid JSON).
        """
        current_time = time.time()
        full_api_url = self.mcp_so_url + self.api_endpoint

        # Check cache validity
        if self._cached_services is not None and \
           (current_time - self._last_fetch_time) < self.cache_duration_seconds:
            logger.info(f"Cache hit for available services from {full_api_url}. Returning cached data.")
            return self._cached_services

        logger.info(f"Cache miss or expired. Fetching available services from {full_api_url}.")
        # Cache is invalid or non-existent, fetch from the platform
        try:
            response = requests.get(full_api_url, timeout=10) # 10-second timeout
            response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
            logger.debug(f"Successfully fetched data from {full_api_url}, status: {response.status_code}")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} while fetching from {full_api_url}: {e}", exc_info=True)
            raise MCPPlatformUnavailableError(
                url=full_api_url,
                details=f"HTTP error {e.response.status_code} while fetching services: {e}"
            )
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout while fetching from {full_api_url}: {e}", exc_info=True)
            raise MCPPlatformUnavailableError(
                url=full_api_url,
                details="Request timed out while fetching services."
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error while fetching from {full_api_url}: {e}", exc_info=True)
            raise MCPPlatformUnavailableError(
                url=full_api_url,
                details="Connection error while fetching services. Ensure the server is reachable."
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Unexpected request error while fetching from {full_api_url}: {e}", exc_info=True)
            raise MCPPlatformUnavailableError(
                url=full_api_url,
                details=f"An unexpected error occurred during the request: {e}"
            )

        # Parse the JSON response
        try:
            services_data = response.json()
            if not isinstance(services_data, list):
                logger.error(f"Invalid response format from {full_api_url}. Expected a JSON list, got {type(services_data)}.")
                raise MCPServiceRegistryError(
                    f"Invalid response format from {full_api_url}. Expected a JSON list."
                )
            self._cached_services = services_data
            self._last_fetch_time = current_time
            logger.info(f"Successfully retrieved and parsed {len(services_data)} services from {full_api_url}. Updated cache.")
            return self._cached_services

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response from {full_api_url}: {e}", exc_info=True)
            raise MCPServiceRegistryError(
                f"Failed to decode JSON response from {full_api_url}. Details: {e}"
            )
        except MCPServiceRegistryError: # Re-raise if we raised it above for specific format error
            raise
        except Exception as e: # Catch any other unexpected error during parsing or validation
            logger.error(f"Unexpected error processing response from {full_api_url}: {e}", exc_info=True)
            raise MCPServiceRegistryError(
                f"An unexpected error occurred while processing the response from {full_api_url}. Details: {e}"
            )
