import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from unittest.mock import patch, MagicMock
from pathlib import Path
import shutil
import json

# Project specific imports
# Assuming mcp_service_manager is in PYTHONPATH or installed
from mcp_service_manager.api_v1 import router as mcp_api_router
from mcp_service_manager.models import Base as MCPBase # SQLAlchemy Base for MCP models
from mcp_service_manager.models import InstalledMCP
from mcp_service_manager.installer import MCPInstaller
from mcp_service_manager.registry import ServiceRegistry
from mcp_service_manager.schemas import MCPInstallRequest, MCPAvailableListItem, MCPListItem

# --- Test Database Setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:" # In-memory SQLite database

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False} # check_same_thread for SQLite
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Pytest Fixtures ---

@pytest.fixture(scope="function") # Use "function" scope for test isolation
def test_db_session():
    """
    Pytest fixture to set up a test database session.
    Creates all tables before yielding a session and drops them afterwards.
    """
    MCPBase.metadata.create_all(bind=engine) # Create tables
    db = TestingSessionLocal()
    try:
        yield db # Provide the session to the test
    finally:
        db.close()
        MCPBase.metadata.drop_all(bind=engine) # Drop tables after test

@pytest.fixture(scope="function")
def temp_mcp_install_root(tmp_path_factory):
    """
    Creates a temporary root directory for MCP installations for a test session.
    This is different from pytest's tmp_path which is function-scoped.
    We might want a session-scoped one if tests don't interfere, or function-scoped for full isolation.
    Let's use function-scoped for now via tmp_path_factory to ensure clean state.
    """
    return tmp_path_factory.mktemp("mcp_install_root")


@pytest.fixture(scope="function")
def client(test_db_session, temp_mcp_install_root):
    """
    Pytest fixture to create a FastAPI TestClient.
    Overrides the get_db_session dependency for API routes.
    Also configures MCPInstaller's base_install_path to use a temp directory.
    """
    # Override the MCPInstaller's default base path for tests
    # One way is to ensure MCPInstaller instances used by API get this path.
    # This can be complex if MCPInstaller is instantiated deep inside.
    # A simpler way for testing is to patch the DEFAULT_BASE_INSTALL_PATH or
    # ensure the MCPInstaller instance created by the API uses a path we control.
    # For this test, we'll assume api_v1.py might instantiate MCPInstaller directly.
    # The dependency override for get_db_session is more straightforward.

    # Patch the default path directly in the Installer class for the duration of the client fixture
    original_install_path = MCPInstaller.DEFAULT_BASE_INSTALL_PATH
    MCPInstaller.DEFAULT_BASE_INSTALL_PATH = str(temp_mcp_install_root)

    app = FastAPI()
    app.include_router(mcp_api_router)

    # Override the database session dependency
    def override_get_db():
        try:
            yield test_db_session
        finally:
            pass # Session closure is handled by test_db_session fixture

    app.dependency_overrides[mcp_api_router.dependencies[0].dependency] = override_get_db

    yield TestClient(app) # Yield the TestClient

    # Teardown: Restore original path (if needed, though test scope might make it okay)
    MCPInstaller.DEFAULT_BASE_INSTALL_PATH = original_install_path
    # Clean up temp_mcp_install_root if necessary, though tmp_path_factory handles its own cleanup
    # if temp_mcp_install_root.exists():
    #     shutil.rmtree(temp_mcp_install_root)


# --- Mock Data ---
MOCK_AVAILABLE_SERVICES_DATA = [
    {
        "id": "mock-mcp-alpha",
        "name": "Mock MCP Alpha",
        "description": "A mock MCP for testing.",
        "version": "1.0.0",
        "download_url": "https://example.com/mock-mcp-alpha-1.0.0.zip"
    },
    {
        "id": "mock-mcp-beta",
        "name": "Mock MCP Beta",
        "description": "Another mock MCP for testing.",
        "version": "0.9.0",
        "download_url": "https://example.com/mock-mcp-beta-0.9.0.tar.gz"
    }
]

# --- Integration Tests ---

@patch.object(ServiceRegistry, 'list_available_services')
@patch.object(MCPInstaller, '_download_package')
@patch.object(MCPInstaller, '_extract_package')
def test_mcp_full_lifecycle(
    mock_extract, mock_download, mock_list_available,
    client: TestClient, test_db_session: Session, temp_mcp_install_root: Path
):
    """
    Tests a typical lifecycle: list available, install, check details, update config,
    enable/disable, and uninstall.
    """
    # --- 1. GET /available ---
    mock_list_available.return_value = MOCK_AVAILABLE_SERVICES_DATA
    response = client.get("/api/v1/mcp/available")
    assert response.status_code == 200
    available_services = response.json()
    assert available_services == MOCK_AVAILABLE_SERVICES_DATA
    mock_list_available.assert_called_once()

    # Select one service to install
    service_to_install = MCPAvailableListItem(**available_services[0])
    mcp_name = service_to_install.name # Use the name from the available service
    mcp_version = service_to_install.version

    # --- 2. POST /install ---
    # Configure mocks for download and extract
    dummy_downloaded_file = temp_mcp_install_root / mcp_name / "temp_download" / f"{mcp_name}_package.zip"
    dummy_downloaded_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_downloaded_file.touch()
    mock_download.return_value = dummy_downloaded_file
    mock_extract.return_value = None # Simulate successful extraction

    install_payload = MCPInstallRequest(
        mcp_name=mcp_name,
        version=mcp_version,
        source_url=f"https://example.com/git/{service_to_install.id}.git", # Construct a plausible source_url
        download_url=service_to_install.download_url,
        config_template={"initial_setting": "default_value"}
    )

    response = client.post("/api/v1/mcp/install", json=install_payload.model_dump())
    assert response.status_code == 201, response.text
    installed_mcp_data = response.json()

    assert installed_mcp_data["mcp_name"] == mcp_name
    assert installed_mcp_data["mcp_version"] == mcp_version
    assert installed_mcp_data["is_enabled"] is True

    # Verify mocks
    mock_download.assert_called_once_with(
        install_payload.download_url,
        Path(MCPInstaller.DEFAULT_BASE_INSTALL_PATH) / mcp_name / "temp_download", # Path used by installer
        mcp_name
    )
    mcp_content_install_dir = Path(MCPInstaller.DEFAULT_BASE_INSTALL_PATH) / mcp_name / mcp_version / MCPInstaller.MCP_CONTENT_DIR_NAME
    mock_extract.assert_called_once_with(dummy_downloaded_file, mcp_content_install_dir)

    # Verify database record
    db_record = test_db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).first()
    assert db_record is not None
    assert db_record.mcp_name == mcp_name
    assert db_record.mcp_version == mcp_version
    assert db_record.is_enabled is True

    # Verify file/directory structure
    expected_mcp_base_dir = temp_mcp_install_root / mcp_name
    expected_version_dir = expected_mcp_base_dir / mcp_version
    expected_content_dir = expected_version_dir / MCPInstaller.MCP_CONTENT_DIR_NAME
    expected_config_file = expected_mcp_base_dir / MCPInstaller.MCP_CONFIG_FILE_NAME

    assert expected_mcp_base_dir.is_dir()
    assert expected_version_dir.is_dir()
    assert expected_content_dir.is_dir() # Created by install_mcp for extraction
    assert expected_config_file.is_file()
    with open(expected_config_file, 'r') as f:
        config_on_disk = json.load(f)
    assert config_on_disk == install_payload.config_template

    # --- TODO: Implement remaining lifecycle steps ---
    # GET /installed
    # GET /{mcp_name}
    # PUT /{mcp_name}/config
    # POST /{mcp_name}/enable (or disable if already enabled)
    # POST /{mcp_name}/disable
    # DELETE /{mcp_name}

    # --- 3. GET /installed ---
    response = client.get("/api/v1/mcp/installed")
    assert response.status_code == 200
    installed_list = response.json()
    assert len(installed_list) == 1
    assert installed_list[0]["mcp_name"] == mcp_name
    assert installed_list[0]["mcp_version"] == mcp_version

    # --- 4. GET /{mcp_name} ---
    response = client.get(f"/api/v1/mcp/{mcp_name}")
    assert response.status_code == 200
    details = response.json()
    assert details["mcp_name"] == mcp_name
    assert details["is_enabled"] is True
    assert Path(details["local_path"]) == expected_content_dir.resolve()
    assert Path(details["config_file_path"]) == expected_config_file.resolve()

    # --- 5. PUT /{mcp_name}/config ---
    new_config_data = {"updated_setting": "new_value", "port": 9090}
    response = client.put(f"/api/v1/mcp/{mcp_name}/config", json={"config_json": new_config_data})
    assert response.status_code == 200, response.text
    updated_mcp_data = response.json()
    assert updated_mcp_data["mcp_name"] == mcp_name # API returns the MCPListItem

    # Verify config file content on disk
    with open(expected_config_file, 'r') as f:
        config_on_disk_after_update = json.load(f)
    assert config_on_disk_after_update == new_config_data
    # Note: The API currently doesn't update `last_updated` on config change. If it did, verify here.

    # --- 6. POST /{mcp_name}/disable ---
    response = client.post(f"/api/v1/mcp/{mcp_name}/disable")
    assert response.status_code == 200, response.text
    disabled_mcp_data = response.json()
    assert disabled_mcp_data["is_enabled"] is False

    # Verify DB status
    test_db_session.refresh(db_record) # Refresh from DB
    assert db_record.is_enabled is False

    # --- 7. POST /{mcp_name}/enable ---
    response = client.post(f"/api/v1/mcp/{mcp_name}/enable")
    assert response.status_code == 200, response.text
    enabled_mcp_data = response.json()
    assert enabled_mcp_data["is_enabled"] is True

    # Verify DB status
    test_db_session.refresh(db_record) # Refresh from DB
    assert db_record.is_enabled is True

    # --- 8. DELETE /{mcp_name} ---
    response = client.delete(f"/api/v1/mcp/{mcp_name}")
    assert response.status_code == 204, response.text # No content on successful delete

    # Verify removed from DB
    db_record_after_delete = test_db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).first()
    assert db_record_after_delete is None

    # Verify files/directories are cleaned up
    # The entire base directory for the MCP should be gone.
    assert not expected_mcp_base_dir.exists()

@patch.object(ServiceRegistry, 'list_available_services')
@patch.object(MCPInstaller, '_download_package')
@patch.object(MCPInstaller, '_extract_package')
def test_install_mcp_already_exists(
    mock_extract, mock_download, mock_list_available,
    client: TestClient, test_db_session: Session, temp_mcp_install_root: Path
):
    """Test attempting to install an MCP that already exists."""
    # --- Arrange: First installation (successful) ---
    mock_list_available.return_value = MOCK_AVAILABLE_SERVICES_DATA
    service_to_install = MCPAvailableListItem(**MOCK_AVAILABLE_SERVICES_DATA[0])
    mcp_name = service_to_install.name
    mcp_version = service_to_install.version

    dummy_downloaded_file = temp_mcp_install_root / mcp_name / "temp_download" / f"{mcp_name}_package.zip"
    dummy_downloaded_file.parent.mkdir(parents=True, exist_ok=True)
    dummy_downloaded_file.touch()
    mock_download.return_value = dummy_downloaded_file
    mock_extract.return_value = None

    install_payload = MCPInstallRequest(
        mcp_name=mcp_name, version=mcp_version,
        source_url="https://example.com/git/first.git",
        download_url=service_to_install.download_url,
        config_template={"original": "config"}
    )
    response_first_install = client.post("/api/v1/mcp/install", json=install_payload.model_dump())
    assert response_first_install.status_code == 201

    # Reset mocks for the second call if necessary, though for different error it might not matter
    mock_download.reset_mock()
    mock_extract.reset_mock()
    # mock_db_session.reset_mock() # Be careful with resetting session mock if it holds state needed

    # --- Act: Attempt second installation ---
    install_payload_again = MCPInstallRequest( # Same name and version
        mcp_name=mcp_name, version=mcp_version,
        source_url="https://example.com/git/second.git", # Different source/download to ensure it's a new attempt
        download_url="https://example.com/new_download.zip",
        config_template={"new": "config"}
    )
    response_second_install = client.post("/api/v1/mcp/install", json=install_payload_again.model_dump())

    # --- Assert ---
    assert response_second_install.status_code == 409 # Conflict
    detail = response_second_install.json().get("detail", "")
    assert f"MCP '{mcp_name}' is already installed." in detail

    # Ensure no new download/extract happened for the second attempt
    mock_download.assert_not_called()
    mock_extract.assert_not_called()

    # Ensure only one record in DB
    count = test_db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).count()
    assert count == 1
