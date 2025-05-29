import pytest
import json
from pathlib import Path

# Assuming the mcp_service_manager package is installed or in PYTHONPATH
# If running pytest from the root of the project, Python should find it.
# Adjust if your project structure requires specific path manipulation for tests.
from mcp_service_manager.core import MCPConfig
from mcp_service_manager.exceptions import MCPFileNotFoundError, MCPJSONDecodeError

@pytest.fixture
def valid_config_data():
    return {"key1": "value1", "key2": {"nested_key": "nested_value"}, "port": 8080}

@pytest.fixture
def temp_config_file(tmp_path, valid_config_data) -> Path:
    config_file = tmp_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(valid_config_data, f)
    return config_file

@pytest.fixture
def temp_invalid_json_file(tmp_path) -> Path:
    config_file = tmp_path / "invalid_config.json"
    with open(config_file, 'w') as f:
        f.write("{'key': 'value',") # Invalid JSON (missing closing brace, single quotes)
    return config_file

def test_mcp_config_load_success(temp_config_file, valid_config_data):
    """Test successful loading of a valid JSON config file."""
    config = MCPConfig(config_file_path=str(temp_config_file))
    config.load_config()
    assert config.config_data == valid_config_data

def test_mcp_config_file_not_found(tmp_path):
    """Test MCPFileNotFoundError when config file does not exist."""
    non_existent_file = tmp_path / "non_existent_config.json"
    config = MCPConfig(config_file_path=str(non_existent_file))
    with pytest.raises(MCPFileNotFoundError) as excinfo:
        config.load_config()
    assert str(non_existent_file) in str(excinfo.value)

def test_mcp_config_invalid_json(temp_invalid_json_file):
    """Test MCPJSONDecodeError when config file contains invalid JSON."""
    config = MCPConfig(config_file_path=str(temp_invalid_json_file))
    with pytest.raises(MCPJSONDecodeError) as excinfo:
        config.load_config()
    assert str(temp_invalid_json_file) in str(excinfo.value)
    assert "Error decoding JSON" in str(excinfo.value)

def test_mcp_config_get_existing_key(temp_config_file, valid_config_data):
    """Test get() method for an existing key."""
    config = MCPConfig(config_file_path=str(temp_config_file))
    config.load_config()
    assert config.get("key1") == valid_config_data["key1"]
    assert config.get("port") == valid_config_data["port"]
    assert config.get("key2") == valid_config_data["key2"]

def test_mcp_config_get_non_existing_key(temp_config_file):
    """Test get() method for a non-existing key (should return None)."""
    config = MCPConfig(config_file_path=str(temp_config_file))
    config.load_config()
    assert config.get("non_existent_key") is None

def test_mcp_config_get_non_existing_key_with_default(temp_config_file):
    """Test get() method for a non-existing key with a default value."""
    config = MCPConfig(config_file_path=str(temp_config_file))
    config.load_config()
    default_val = "default_value"
    assert config.get("non_existent_key", default=default_val) == default_val

def test_mcp_config_get_before_load(tmp_path):
    """Test get() method before load_config() is called."""
    config = MCPConfig(config_file_path=str(tmp_path / "dummy.json"))
    assert config.get("any_key") is None
    assert config.get("any_key", default="default") == "default"


def test_mcp_config_get_all_success(temp_config_file, valid_config_data):
    """Test get_all() method after successful load."""
    config = MCPConfig(config_file_path=str(temp_config_file))
    config.load_config()
    assert config.get_all() == valid_config_data

def test_mcp_config_get_all_before_load(tmp_path):
    """Test get_all() method before load_config() is called (should return empty dict)."""
    config = MCPConfig(config_file_path=str(tmp_path / "dummy.json"))
    assert config.get_all() == {}

def test_mcp_config_get_all_after_failed_load(temp_invalid_json_file):
    """Test get_all() method after a failed load_config() (should return empty dict)."""
    config = MCPConfig(config_file_path=str(temp_invalid_json_file))
    with pytest.raises(MCPJSONDecodeError):
        config.load_config()
    # Depending on implementation, config_data might be None or {} after error.
    # The current MCPConfig implementation sets self.config_data only on success.
    # So, if load_config fails, self.config_data remains None.
    assert config.get_all() == {}
    assert config.config_data is None

def test_mcp_config_load_config_overwrites_previous(tmp_path, valid_config_data):
    """Test that calling load_config again reloads and overwrites previous data."""
    config_file_v1 = tmp_path / "config_v1.json"
    config_file_v2 = tmp_path / "config_v2.json"

    data_v1 = {"version": 1, "setting": "alpha"}
    data_v2 = {"version": 2, "setting": "beta", "new_field": True}

    with open(config_file_v1, 'w') as f:
        json.dump(data_v1, f)
    with open(config_file_v2, 'w') as f:
        json.dump(data_v2, f)

    # Load v1
    config = MCPConfig(config_file_path=str(config_file_v1))
    config.load_config()
    assert config.get_all() == data_v1

    # "Point" to v2 and reload (in a real scenario, filepath might not change, but content would)
    # For this test, we change the filepath attribute of the config object.
    config.config_file_path = str(config_file_v2)
    config.load_config()
    assert config.get_all() == data_v2
    assert config.get("version") == 2
    assert config.get("new_field") is True
    assert config.get("setting") == "beta"
