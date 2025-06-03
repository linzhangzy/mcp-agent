import pytest
from unittest.mock import MagicMock, patch, mock_open
import sys
import json
from pathlib import Path
import importlib

# Project specific imports
from mcp_service_manager.loader import PluginLoader
from mcp_service_manager.models import InstalledMCP
from mcp_service_manager.core import MCPConfig, AbstractMCPPlugin
from mcp_service_manager.exceptions import (
    MCPLoadError,
    MCPManifestError,
    MCPModuleNotFoundError,
    MCPClassNotFoundError,
    MCPPluginInitializationError,
    MCPFileNotFoundError,
    MCPJSONDecodeError
)

# --- Fixtures ---

@pytest.fixture
def plugin_loader_instance():
    """Provides a PluginLoader instance."""
    return PluginLoader()

@pytest.fixture
def sample_mcp_record(tmp_path: Path) -> InstalledMCP:
    """Provides a sample InstalledMCP object with paths in a temporary directory."""
    mcp_name = "TestMCP"
    mcp_root_path = tmp_path / mcp_name
    mcp_root_path.mkdir(parents=True, exist_ok=True)

    # Create a dummy config file for MCPConfig to load
    config_file = mcp_root_path / "mcp_config.json"
    with open(config_file, 'w') as f:
        json.dump({"setting1": "value1"}, f)

    record = MagicMock(spec=InstalledMCP)
    record.mcp_name = mcp_name
    record.local_path = str(mcp_root_path) # This is where manifest & plugin code will reside
    record.config_file_path = str(config_file)
    return record

@pytest.fixture
def dummy_plugin_code_path(tmp_path: Path) -> Path:
    """Creates a directory for dummy plugin code and returns its path."""
    code_path = tmp_path / "dummy_plugin_src"
    code_path.mkdir(parents=True, exist_ok=True)
    return code_path

# --- Dummy Plugin Implementation for Testing ---
# This string will be written to a .py file in tests
DUMMY_PLUGIN_PY_CONTENT = """
from mcp_service_manager.core import AbstractMCPPlugin, MCPConfig # Assuming this is findable
import logging

logger = logging.getLogger(__name__)

class DummyPlugin(AbstractMCPPlugin):
    def __init__(self, mcp_config: MCPConfig):
        super().__init__(mcp_config)
        self.initialized_with_config = mcp_config
        self.config_value = self.mcp_config.get("plugin_specific_setting", "default_plugin_value")
        logger.info(f"DummyPlugin initialized with config value: {self.config_value}")

    def get_status(self) -> dict:
        return {"status": "ok", "config_value": self.config_value}

    def on_enable(self):
        logger.info("DummyPlugin on_enable called")

    def on_disable(self):
        logger.info("DummyPlugin on_disable called")

class NotAPlugin: # For testing wrong class type
    def __init__(self, mcp_config: MCPConfig):
        pass

class InitFailsPlugin(AbstractMCPPlugin):
    def __init__(self, mcp_config: MCPConfig):
        super().__init__(mcp_config)
        raise ValueError("Plugin initialization failed deliberately")
    def get_status(self) -> dict: return {}
    def on_enable(self): pass
    def on_disable(self): pass
"""

# --- Test Cases ---

def test_load_mcp_plugin_success(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP, dummy_plugin_code_path: Path):
    """Test successful loading of a valid plugin."""
    # Setup: Create manifest and dummy plugin module inside the sample_mcp_record.local_path
    mcp_root_path = Path(sample_mcp_record.local_path)

    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "DummyPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)

    # Write the dummy plugin code to dummy_plugin_module.py within mcp_root_path
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT)

    # Also create the config file that the DummyPlugin expects
    plugin_specific_config_data = {"plugin_specific_setting": "test_value_for_plugin"}
    with open(sample_mcp_record.config_file_path, 'w') as f: # Overwrite if already created by fixture
        json.dump(plugin_specific_config_data, f)

    # Action
    plugin_instance = plugin_loader_instance.load_mcp_plugin(sample_mcp_record)

    # Assertions
    assert plugin_instance is not None
    assert isinstance(plugin_instance, AbstractMCPPlugin)
    assert plugin_instance.__class__.__name__ == "DummyPlugin"
    assert hasattr(plugin_instance, 'initialized_with_config')
    assert isinstance(plugin_instance.initialized_with_config, MCPConfig)
    assert plugin_instance.config_value == "test_value_for_plugin"


def test_load_mcp_plugin_manifest_not_found(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPManifestError when mcp_manifest.json is not found."""
    with pytest.raises(MCPManifestError, match="Manifest file .* not found"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)

@patch('builtins.open', new_callable=mock_open, read_data="this is not json")
def test_load_mcp_plugin_manifest_invalid_json(mock_file_open, plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPManifestError for malformed JSON in manifest."""
    # Ensure manifest file appears to exist, but content is bad
    Path(sample_mcp_record.local_path, PluginLoader.MANIFEST_FILE_NAME).touch()

    with pytest.raises(MCPManifestError, match="Invalid JSON in manifest file"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)

@pytest.mark.parametrize("manifest_content, error_detail", [
    ({}, "must contain an 'entry_point' object"),
    ({"entry_point": "not_a_dict"}, "must contain an 'entry_point' object"),
    ({"entry_point": {"class": "MyPlugin"}}, "with 'module' and 'class' keys"), # Missing module
    ({"entry_point": {"module": "my_module"}}, "with 'module' and 'class' keys"), # Missing class
])
def test_load_mcp_plugin_manifest_missing_keys(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP, manifest_content: dict, error_detail: str):
    """Test MCPManifestError for missing required keys in manifest."""
    manifest_path = Path(sample_mcp_record.local_path) / PluginLoader.MANIFEST_FILE_NAME
    with open(manifest_path, 'w') as f:
        json.dump(manifest_content, f)

    with pytest.raises(MCPManifestError, match=error_detail):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)


def test_load_mcp_plugin_module_not_found(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPModuleNotFoundError when the module specified in manifest doesn't exist."""
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "non_existent_module", "class": "AnyClass"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)

    with pytest.raises(MCPModuleNotFoundError, match="Failed to import module"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)


def test_load_mcp_plugin_class_not_found(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPClassNotFoundError when class specified in manifest doesn't exist in module."""
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "NonExistentClass"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)

    # Create the dummy module but without the NonExistentClass
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT) # DummyPlugin is in here, but not NonExistentClass

    with pytest.raises(MCPClassNotFoundError, match="Class .* not found in module"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)


def test_load_mcp_plugin_initialization_fails(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPPluginInitializationError when plugin's __init__ fails."""
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "InitFailsPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT)

    with pytest.raises(MCPPluginInitializationError, match="Error during instantiation of plugin"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)


def test_load_mcp_plugin_not_subclass_of_abstract(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPLoadError if the loaded class is not a subclass of AbstractMCPPlugin."""
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "NotAPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT)

    with pytest.raises(MCPLoadError, match="does not implement AbstractMCPPlugin"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)


def test_load_mcp_plugin_config_load_fails(plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP):
    """Test MCPPluginInitializationError if MCPConfig fails to load."""
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "DummyPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT)

    # Make config file non-existent to cause MCPConfig.load_config() to fail
    Path(sample_mcp_record.config_file_path).unlink(missing_ok=True)

    with pytest.raises(MCPPluginInitializationError, match="Failed to load MCPConfig for plugin"):
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)

@patch('sys.path', new_callable=list) # Patch sys.path to be an empty list initially for this test
def test_sys_path_management(mock_sys_path, plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP, dummy_plugin_code_path: Path):
    """Test that sys.path is correctly managed."""
    mcp_root_path = Path(sample_mcp_record.local_path) # This path comes from tmp_path via fixture

    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "DummyPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)
    with open(mcp_root_path / "dummy_plugin_module.py", 'w') as f:
        f.write(DUMMY_PLUGIN_PY_CONTENT)

    # Mock importlib.import_module to check sys.path state *during* import attempt
    original_import_module = importlib.import_module
    str_mcp_local_path_resolved = str(mcp_root_path.resolve())

    def import_module_side_effect(name):
        # Check that the path is in sys.path when import_module is called
        assert str_mcp_local_path_resolved in sys.path
        # Allow actual import to proceed from the modified sys.path
        # Need to remove our mock temporarily to avoid recursion if module uses importlib
        with patch('importlib.import_module', original_import_module):
             # If the module is already in sys.modules due to previous test runs or other reasons,
             # import_module might not re-evaluate its path logic in the same way.
             # For a clean test, ensure it's not pre-loaded from an unexpected location.
            if name in sys.modules:
                del sys.modules[name]
            return original_import_module(name)

    with patch('importlib.import_module', side_effect=import_module_side_effect) as mock_import:
        # Ensure path is not there before call (mock_sys_path is empty list here)
        assert str_mcp_local_path_resolved not in mock_sys_path

        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)

        mock_import.assert_called_with("dummy_plugin_module")

    # Ensure path is removed after call
    assert str_mcp_local_path_resolved not in sys.path
    # Also check our mock_sys_path (though the real sys.path is what matters for cleanup)
    assert str_mcp_local_path_resolved not in mock_sys_path


# Test for config file not found during MCPConfig.load_config() within plugin loader
@patch.object(MCPConfig, 'load_config', side_effect=MCPFileNotFoundError("dummy/path/mcp_config.json"))
def test_load_mcp_plugin_config_file_not_found_error(
    mock_load_config, plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP
):
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "DummyPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)
    # No need to create dummy_plugin_module.py if MCPConfig load fails first

    with pytest.raises(MCPPluginInitializationError) as excinfo:
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)
    assert "Failed to load MCPConfig for plugin" in str(excinfo.value)
    assert "dummy/path/mcp_config.json" in str(excinfo.value)


@patch.object(MCPConfig, 'load_config', side_effect=MCPJSONDecodeError("dummy/path/mcp_config.json", Exception("json error")))
def test_load_mcp_plugin_config_json_decode_error(
    mock_load_config, plugin_loader_instance: PluginLoader, sample_mcp_record: InstalledMCP
):
    mcp_root_path = Path(sample_mcp_record.local_path)
    manifest_content = {"entry_point": {"module": "dummy_plugin_module", "class": "DummyPlugin"}}
    with open(mcp_root_path / PluginLoader.MANIFEST_FILE_NAME, 'w') as f:
        json.dump(manifest_content, f)

    with pytest.raises(MCPPluginInitializationError) as excinfo:
        plugin_loader_instance.load_mcp_plugin(sample_mcp_record)
    assert "Failed to load MCPConfig for plugin" in str(excinfo.value)
    assert "json error" in str(excinfo.value)
