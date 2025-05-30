import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

# Project specific imports
from mcp_service_manager.monitor import ServiceMonitor
from mcp_service_manager.models import InstalledMCP
from mcp_service_manager.exceptions import MCPNotFoundError

# --- Fixtures ---

@pytest.fixture
def mock_db_session():
    """Provides a MagicMock for the SQLAlchemy session."""
    session = MagicMock(spec=Session) # Use spec=Session if Session is importable and useful
    # Mock the query chain for filter_by(...).first()
    session.query.return_value.filter_by.return_value.first.return_value = None
    # Mock the query chain for all()
    session.query.return_value.all.return_value = []
    return session

@pytest.fixture
def service_monitor_instance(mock_db_session):
    """Provides a ServiceMonitor instance with a mocked db session."""
    return ServiceMonitor(db_session=mock_db_session)

@pytest.fixture
def sample_mcp_record_1():
    """Provides a sample InstalledMCP-like object for tests."""
    record = MagicMock(spec=InstalledMCP)
    record.id = 1
    record.mcp_name = "TestMCP_Alpha"
    record.mcp_version = "1.0.1"
    record.is_enabled = True
    record.installed_at = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    record.last_updated = datetime(2023, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
    record.local_path = "/mnt/mcps/TestMCP_Alpha/1.0.1/mcp_content"
    record.config_file_path = "/mnt/mcps/TestMCP_Alpha/mcp_config.json"
    return record

@pytest.fixture
def sample_mcp_record_2():
    """Provides another sample InstalledMCP-like object for tests."""
    record = MagicMock(spec=InstalledMCP)
    record.id = 2
    record.mcp_name = "TestMCP_Beta"
    record.mcp_version = "0.9.5"
    record.is_enabled = False
    record.installed_at = datetime(2023, 2, 10, 8, 0, 0, tzinfo=timezone.utc)
    record.last_updated = datetime(2023, 2, 20, 18, 0, 0, tzinfo=timezone.utc)
    record.local_path = "/mnt/mcps/TestMCP_Beta/0.9.5/mcp_content"
    record.config_file_path = "/mnt/mcps/TestMCP_Beta/mcp_config.json"
    return record

# --- Test Cases for get_mcp_status ---

def test_get_mcp_status_success(service_monitor_instance: ServiceMonitor, mock_db_session: MagicMock, sample_mcp_record_1: MagicMock):
    """Test successful retrieval of status for an existing MCP."""
    # Configure mock_db_session to return the sample record
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = sample_mcp_record_1

    status = service_monitor_instance.get_mcp_status(mcp_name="TestMCP_Alpha")

    mock_db_session.query.assert_called_once_with(InstalledMCP)
    mock_db_session.query.return_value.filter_by.assert_called_once_with(mcp_name="TestMCP_Alpha")
    
    assert status["name"] == sample_mcp_record_1.mcp_name
    assert status["version"] == sample_mcp_record_1.mcp_version
    assert status["is_enabled"] == sample_mcp_record_1.is_enabled
    assert status["installed_at"] == sample_mcp_record_1.installed_at.isoformat()
    assert status["last_updated"] == sample_mcp_record_1.last_updated.isoformat()
    assert status["local_path"] == sample_mcp_record_1.local_path
    assert status["config_file_path"] == sample_mcp_record_1.config_file_path

def test_get_mcp_status_not_found(service_monitor_instance: ServiceMonitor, mock_db_session: MagicMock):
    """Test MCPNotFoundError when the MCP is not found."""
    # Configure mock_db_session to return None (default for the fixture, but explicit here for clarity)
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = None

    with pytest.raises(MCPNotFoundError, match="MCP 'NonExistentMCP' not found."):
        service_monitor_instance.get_mcp_status(mcp_name="NonExistentMCP")
    
    mock_db_session.query.assert_called_once_with(InstalledMCP)
    mock_db_session.query.return_value.filter_by.assert_called_once_with(mcp_name="NonExistentMCP")


# --- Test Cases for list_all_mcp_statuses ---

def test_list_all_mcp_statuses_multiple_mcps(
    service_monitor_instance: ServiceMonitor, 
    mock_db_session: MagicMock, 
    sample_mcp_record_1: MagicMock, 
    sample_mcp_record_2: MagicMock
):
    """Test listing statuses when multiple MCPs are installed."""
    mock_db_session.query.return_value.all.return_value = [sample_mcp_record_1, sample_mcp_record_2]

    statuses = service_monitor_instance.list_all_mcp_statuses()

    mock_db_session.query.assert_called_once_with(InstalledMCP)
    assert len(statuses) == 2

    # Verify content of the first record's status
    assert statuses[0]["name"] == sample_mcp_record_1.mcp_name
    assert statuses[0]["version"] == sample_mcp_record_1.mcp_version
    assert statuses[0]["is_enabled"] == sample_mcp_record_1.is_enabled
    assert statuses[0]["installed_at"] == sample_mcp_record_1.installed_at.isoformat()

    # Verify content of the second record's status
    assert statuses[1]["name"] == sample_mcp_record_2.mcp_name
    assert statuses[1]["version"] == sample_mcp_record_2.mcp_version
    assert statuses[1]["is_enabled"] == sample_mcp_record_2.is_enabled
    assert statuses[1]["installed_at"] == sample_mcp_record_2.installed_at.isoformat()


def test_list_all_mcp_statuses_no_mcps(service_monitor_instance: ServiceMonitor, mock_db_session: MagicMock):
    """Test listing statuses when no MCPs are installed."""
    # Configure mock_db_session to return an empty list (default for the fixture)
    mock_db_session.query.return_value.all.return_value = []

    statuses = service_monitor_instance.list_all_mcp_statuses()

    mock_db_session.query.assert_called_once_with(InstalledMCP)
    assert len(statuses) == 0
    assert statuses == []

def test_get_mcp_status_timestamps_can_be_none(service_monitor_instance: ServiceMonitor, mock_db_session: MagicMock, sample_mcp_record_1: MagicMock):
    """Test that status is correctly reported if timestamps are None in DB (although default is func.now())."""
    sample_mcp_record_1.installed_at = None
    sample_mcp_record_1.last_updated = None
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = sample_mcp_record_1

    status = service_monitor_instance.get_mcp_status(mcp_name="TestMCP_Alpha")
    
    assert status["installed_at"] is None
    assert status["last_updated"] is None

def test_list_all_mcp_statuses_timestamps_can_be_none(
    service_monitor_instance: ServiceMonitor, 
    mock_db_session: MagicMock, 
    sample_mcp_record_1: MagicMock
):
    """Test listing with MCPs that might have None timestamps."""
    sample_mcp_record_1.installed_at = None
    sample_mcp_record_1.last_updated = None
    mock_db_session.query.return_value.all.return_value = [sample_mcp_record_1]

    statuses = service_monitor_instance.list_all_mcp_statuses()
    
    assert len(statuses) == 1
    assert statuses[0]["installed_at"] is None
    assert statuses[0]["last_updated"] is None

# End of tests/unit/test_service_monitor.py
