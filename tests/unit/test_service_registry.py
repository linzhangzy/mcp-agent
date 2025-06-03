import pytest
import requests_mock
import time
import json
from unittest.mock import patch

# Assuming the mcp_service_manager package is installed or in PYTHONPATH
from mcp_service_manager.registry import ServiceRegistry
from mcp_service_manager.exceptions import MCPPlatformUnavailableError, MCPServiceRegistryError

MOCK_MCP_SO_URL = "https://mcp.so/" # Ensure this matches the default or is configurable
MOCK_API_ENDPOINT = ServiceRegistry.DEFAULT_API_ENDPOINT # "api/servers"
FULL_MOCK_URL = MOCK_MCP_SO_URL + MOCK_API_ENDPOINT

@pytest.fixture
def mock_services_data():
    return [
        {"id": "service1", "name": "MCP Service One", "version": "1.0"},
        {"id": "service2", "name": "MCP Service Two", "version": "2.1"},
    ]

def test_service_registry_init_url_formatting():
    """Test that the mcp_so_url is correctly formatted with a trailing slash."""
    registry_no_slash = ServiceRegistry(mcp_so_url="https://mcp.so")
    assert registry_no_slash.mcp_so_url == "https://mcp.so/"
    registry_with_slash = ServiceRegistry(mcp_so_url="https://mcp.so/")
    assert registry_with_slash.mcp_so_url == "https://mcp.so/"


def test_list_available_services_success(mock_services_data):
    """Test successful fetching and parsing of MCP services list."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, json=mock_services_data, status_code=200)
        services = registry.list_available_services()
        assert services == mock_services_data
        assert m.called_once # Ensure the mock was actually called

def test_list_available_services_caching(mock_services_data):
    """Test caching logic for list_available_services."""
    cache_duration = 1  # 1 second for quick testing
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL, cache_duration_seconds=cache_duration)

    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, json=mock_services_data, status_code=200)

        # First call - should fetch from mock
        services1 = registry.list_available_services()
        assert services1 == mock_services_data
        assert m.call_count == 1
        assert registry._cached_services == mock_services_data
        last_fetch_time_1 = registry._last_fetch_time

        # Second call - within cache duration, should use cache
        time.sleep(cache_duration / 2) # Sleep for less than cache duration
        services2 = registry.list_available_services()
        assert services2 == mock_services_data
        assert m.call_count == 1 # Mock should still only be called once
        assert registry._last_fetch_time == last_fetch_time_1 # Fetch time should not change

        # Third call - after cache expiry, should fetch from mock again
        time.sleep(cache_duration + 0.1) # Sleep for more than cache duration
        services3 = registry.list_available_services()
        assert services3 == mock_services_data
        assert m.call_count == 2 # Mock should be called again
        assert registry._last_fetch_time > last_fetch_time_1


def test_list_available_services_http_error_404(mock_services_data):
    """Test handling of HTTP 404 error."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, status_code=404, reason="Not Found")
        with pytest.raises(MCPPlatformUnavailableError) as excinfo:
            registry.list_available_services()
        assert "HTTP error 404" in str(excinfo.value)
        assert "Not Found" in str(excinfo.value)

def test_list_available_services_http_error_500():
    """Test handling of HTTP 500 error."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, status_code=500, text="Internal Server Error")
        with pytest.raises(MCPPlatformUnavailableError) as excinfo:
            registry.list_available_services()
        assert "HTTP error 500" in str(excinfo.value)

def test_list_available_services_connection_error():
    """Test handling of requests.exceptions.ConnectionError."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        import requests # for requests.exceptions.ConnectionError
        m.get(FULL_MOCK_URL, exc=requests.exceptions.ConnectionError("Failed to connect"))
        with pytest.raises(MCPPlatformUnavailableError) as excinfo:
            registry.list_available_services()
        assert "Connection error" in str(excinfo.value)

def test_list_available_services_timeout_error():
    """Test handling of requests.exceptions.Timeout."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        import requests # for requests.exceptions.Timeout
        m.get(FULL_MOCK_URL, exc=requests.exceptions.Timeout("Request timed out"))
        with pytest.raises(MCPPlatformUnavailableError) as excinfo:
            registry.list_available_services()
        assert "Request timed out" in str(excinfo.value)

def test_list_available_services_invalid_json_response():
    """Test handling of invalid JSON response from the platform."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, text="This is not JSON", status_code=200)
        with pytest.raises(MCPServiceRegistryError) as excinfo:
            registry.list_available_services()
        assert "Failed to decode JSON response" in str(excinfo.value)

def test_list_available_services_non_list_json_response(mock_services_data):
    """Test handling of JSON response that is not a list (e.g. a dict)."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        # Simulate a valid JSON but not a list as expected
        m.get(FULL_MOCK_URL, json={"services": mock_services_data}, status_code=200)
        with pytest.raises(MCPServiceRegistryError) as excinfo:
            registry.list_available_services()
        assert "Invalid response format" in str(excinfo.value)
        assert "Expected a JSON list" in str(excinfo.value)

def test_list_available_services_empty_list_response():
    """Test handling of an empty list response (which is valid)."""
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, json=[], status_code=200)
        services = registry.list_available_services()
        assert services == []

@patch('time.time') # Patch time.time for precise cache control
def test_service_registry_cache_expiry_exact_timing(mock_time, mock_services_data):
    """Test cache expiry with more precise timing control using mock_time."""
    cache_duration = 100
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL, cache_duration_seconds=cache_duration)

    with requests_mock.Mocker() as m:
        m.get(FULL_MOCK_URL, json=mock_services_data, status_code=200)

        # First call
        mock_time.return_value = 1000.0
        services1 = registry.list_available_services()
        assert services1 == mock_services_data
        assert m.call_count == 1

        # Second call, time moved forward but still within cache duration
        mock_time.return_value = 1000.0 + cache_duration - 1
        services2 = registry.list_available_services()
        assert services2 == mock_services_data
        assert m.call_count == 1 # Should use cache

        # Third call, time moved to exactly cache expiry, should still use cache (current_time - last_fetch < duration)
        mock_time.return_value = 1000.0 + cache_duration
        services3 = registry.list_available_services()
        assert services3 == mock_services_data
        # The condition is `(current_time - self._last_fetch_time) < self.cache_duration_seconds`
        # So if current_time - last_fetch == duration, it's not <, so it refetches.
        assert m.call_count == 2 # Should re-fetch

        # Fourth call, time moved past cache expiry
        mock_time.return_value = 1000.0 + cache_duration + 1
        services4 = registry.list_available_services()
        assert services4 == mock_services_data
        # If the previous call refetched at 1100.0, this one might use cache if _last_fetch_time was updated to 1100.0
        # Let's check the call count based on the logic for the third call
        # If the third call refetched, _last_fetch_time is now 1100.0. current_time is 1101.0.
        # 1101.0 - 1100.0 = 1.  1 < 100 is true. So it should use cache.
        assert m.call_count == 2 # Should use cache from the previous refetch.
                                 # The key is that _last_fetch_time was updated in call 3.

    # Verify that time.time was actually called by the registry's list_available_services
    assert mock_time.called


def test_list_available_services_cache_is_none_initially():
    """Test that _cached_services is None initially."""
    registry = ServiceRegistry()
    assert registry._cached_services is None
    assert registry._last_fetch_time == 0.0

# It might be useful to test the full URL construction if the API endpoint could change
def test_service_registry_url_construction_non_default_endpoint():
    """Test URL construction with a non-default API endpoint."""
    custom_endpoint = "api/v2/mcp_services"
    registry = ServiceRegistry(mcp_so_url=MOCK_MCP_SO_URL)
    registry.api_endpoint = custom_endpoint # Manually set for this test

    with requests_mock.Mocker() as m:
        m.get(MOCK_MCP_SO_URL + custom_endpoint, json=[], status_code=200)
        registry.list_available_services()
        assert m.called_once
        assert m.last_request.url == MOCK_MCP_SO_URL + custom_endpoint
