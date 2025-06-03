import json # For PUT /config
from pathlib import Path # For PUT /config
from typing import List, Optional, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session # For type hinting, will be part of get_db

# Project specific imports
from . import (
    ServiceRegistry,
    MCPInstaller,
    ServiceMonitor,
    MCPConfig, # May not be directly used if we write to file directly
    InstalledMCP, # SQLAlchemy model for type hinting
    MCPError, # Base error for generic handling
    MCPNotFoundError,
    MCPAlreadyInstalledError,
    MCPDownloadError,
    MCPExtractionError,
    MCPInstallationError,
    MCPServiceRegistryError,
    MCPPlatformUnavailableError,
    MCPConfigError,
    MCPFileNotFoundError,
    MCPJSONDecodeError
)
from .schemas import (
    MCPInstallRequest,
    MCPConfigUpdateRequest,
    MCPListItem,
    MCPAvailableListItem,
    MCPStatus,
    GeneralResponse
)

logger = logging.getLogger("mcp_manager")

# --- Router Setup ---
router = APIRouter(
    prefix="/api/v1/mcp",
    tags=["MCP Management"],
)

# --- Database Dependency (Placeholder) ---
# This is a simplified placeholder. In a real app, this would be configured
# in main.py or a dedicated database.py with proper session management.
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker
# SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db" # Example, use your actual DB URL
# engine = create_engine(SQLALCHEMY_DATABASE_URL)
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    """
    Placeholder for FastAPI dependency to get a DB session.
    Replace with your actual database session provider.
    Example:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
    """
    # For now, as we don't have a live DB in this environment,
    # we'll return None and services need to handle it or be mocked.
    # In a real scenario, this would yield a SQLAlchemy Session.
    yield None # This MUST be replaced by actual DB session logic

# --- Exception Handlers ---

@router.exception_handler(MCPNotFoundError)
async def mcp_not_found_exception_handler(request, exc: MCPNotFoundError):
    logger.warning(f"MCPNotFoundError caught: {exc} for request: {request.method} {request.url}")
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

@router.exception_handler(MCPAlreadyInstalledError)
async def mcp_already_installed_exception_handler(request, exc: MCPAlreadyInstalledError):
    logger.warning(f"MCPAlreadyInstalledError caught: {exc} for request: {request.method} {request.url}")
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

@router.exception_handler(MCPDownloadError)
async def mcp_download_exception_handler(request, exc: MCPDownloadError):
    logger.error(f"MCPDownloadError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Download failed: {exc}")

@router.exception_handler(MCPExtractionError)
async def mcp_extraction_exception_handler(request, exc: MCPExtractionError):
    logger.error(f"MCPExtractionError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Extraction failed: {exc}")

@router.exception_handler(MCPInstallationError) # More generic installation error
async def mcp_installation_exception_handler(request, exc: MCPInstallationError):
    logger.error(f"MCPInstallationError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Installation error: {exc}")

@router.exception_handler(MCPPlatformUnavailableError)
async def mcp_platform_unavailable_handler(request, exc: MCPPlatformUnavailableError):
    logger.error(f"MCPPlatformUnavailableError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))

@router.exception_handler(MCPServiceRegistryError) # Generic registry error
async def mcp_service_registry_handler(request, exc: MCPServiceRegistryError):
    logger.error(f"MCPServiceRegistryError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

@router.exception_handler(MCPConfigError) # Base for config related errors
async def mcp_config_error_handler(request, exc: MCPConfigError):
    logger.error(f"MCPConfigError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

@router.exception_handler(MCPLoadError) # Base for plugin loading errors
async def mcp_load_error_handler(request, exc: MCPLoadError):
    logger.error(f"MCPLoadError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Plugin loading error: {exc}")

@router.exception_handler(MCPError) # Catch-all for other MCP specific errors
async def mcp_generic_error_handler(request, exc: MCPError):
    logger.error(f"Generic MCPError caught: {exc} for request: {request.method} {request.url}", exc_info=True)
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An MCP related error occurred: {exc}")

# --- API Endpoints ---

@router.get("/available", response_model=List[MCPAvailableListItem])
async def list_available_mcps(
    # In a real app, ServiceRegistry might take config for URL, or be a singleton
):
    """
    Lists MCPs available for installation from the central MCP registry (e.g., mcp.so).
    """
    logger.info("Request received for GET /available")
    registry = ServiceRegistry() # Add mcp_so_url if needed, e.g. from settings
    try:
        services = registry.list_available_services()
        logger.info(f"Successfully retrieved {len(services)} available MCPs.")
        return services
    except MCPPlatformUnavailableError as e:
        # Already logged by ServiceRegistry, re-raise for handler
        raise e
    except MCPServiceRegistryError as e:
        # Already logged by ServiceRegistry, re-raise for handler
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in GET /available: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected server error occurred.")


@router.get("/installed", response_model=List[MCPListItem])
async def list_installed_mcps(db: Session = Depends(get_db_session)):
    """
    Lists all currently installed MCPs.
    """
    logger.info("Request received for GET /installed")
    if db is None: # Placeholder check
        logger.error("Database session not available for GET /installed.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")
    installer = MCPInstaller(db_session=db)
    installed_mcps = installer.list_installed_mcps()
    logger.info(f"Successfully retrieved {len(installed_mcps)} installed MCPs.")
    return installed_mcps


@router.post("/install", response_model=MCPListItem, status_code=status.HTTP_201_CREATED)
async def install_mcp_package(
    request_data: MCPInstallRequest,
    db: Session = Depends(get_db_session)
):
    """
    Installs a new MCP package.
    """
    logger.info(f"Request received for POST /install for MCP: {request_data.mcp_name} version: {request_data.version}")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for POST /install MCP: {request_data.mcp_name}.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")

    installer = MCPInstaller(db_session=db)
    installed_mcp = installer.install_mcp(
        mcp_name=request_data.mcp_name,
        version=request_data.version,
        source_url=request_data.source_url,
        download_url=request_data.download_url,
        config_template=request_data.config_template
    )
    logger.info(f"Successfully installed MCP: {installed_mcp.mcp_name} (ID: {installed_mcp.id})")
    return installed_mcp


@router.get("/{mcp_name}", response_model=MCPStatus)
async def get_mcp_details(mcp_name: str, db: Session = Depends(get_db_session)):
    """
    Retrieves status and details for a specific installed MCP.
    """
    logger.info(f"Request received for GET /{mcp_name}")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for GET /{mcp_name}.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")
    monitor = ServiceMonitor(db_session=db)
    mcp_status = monitor.get_mcp_status(mcp_name=mcp_name)
    logger.info(f"Successfully retrieved details for MCP: {mcp_name}")
    return mcp_status


@router.put("/{mcp_name}/config", response_model=MCPListItem)
async def update_mcp_configuration(
    mcp_name: str,
    config_update: MCPConfigUpdateRequest,
    db: Session = Depends(get_db_session)
):
    """
    Updates the mcp_config.json file for a given MCP.
    """
    logger.info(f"Request received for PUT /{mcp_name}/config")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for PUT /{mcp_name}/config.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")

    installer = MCPInstaller(db_session=db)
    mcp_record = installer.get_installed_mcp(mcp_name=mcp_name)
    # MCPNotFoundError is handled by global handler if mcp_record is None
    if not mcp_record: # Should be caught by handler, but defensive
        logger.warning(f"MCP {mcp_name} not found for config update, but not caught by handler (should not happen).")
        raise MCPNotFoundError(mcp_name=mcp_name)


    config_file_path_str = mcp_record.config_file_path
    if not config_file_path_str:
        logger.error(f"Configuration file path not set for MCP '{mcp_name}' (ID: {mcp_record.id}).")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Configuration file path not set for MCP '{mcp_name}'."
        )

    config_file_path = Path(config_file_path_str)
    logger.info(f"Attempting to update config file {config_file_path} for MCP '{mcp_name}'")

    try:
        config_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file_path, 'w') as f:
            json.dump(config_update.config_json, f, indent=4)
        logger.info(f"Successfully updated config file {config_file_path} for MCP '{mcp_name}'")

        # Note: `last_updated` field is not updated in this version.
        # If it were:
        # from datetime import datetime, timezone
        # mcp_record.last_updated = datetime.now(timezone.utc)
        # db.commit()
        # db.refresh(mcp_record)
        # logger.info(f"Updated last_updated timestamp for MCP '{mcp_name}' (ID: {mcp_record.id})")

    except IOError as e:
        logger.error(f"IOError writing config file {config_file_path} for MCP '{mcp_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write configuration file for '{mcp_name}': {e}"
        )
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error updating config for MCP '{mcp_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while updating config for '{mcp_name}': {e}"
        )

    return mcp_record


@router.post("/{mcp_name}/enable", response_model=MCPListItem)
async def enable_mcp_endpoint(mcp_name: str, db: Session = Depends(get_db_session)):
    """
    Enables a disabled MCP.
    """
    logger.info(f"Request received for POST /{mcp_name}/enable")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for POST /{mcp_name}/enable.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")
    installer = MCPInstaller(db_session=db)
    updated_mcp = installer.enable_mcp(mcp_name=mcp_name)
    logger.info(f"Successfully processed enable request for MCP: {mcp_name}. Current status: {'enabled' if updated_mcp.is_enabled else 'still disabled'}")
    return updated_mcp


@router.post("/{mcp_name}/disable", response_model=MCPListItem)
async def disable_mcp_endpoint(mcp_name: str, db: Session = Depends(get_db_session)):
    """
    Disables an enabled MCP.
    """
    logger.info(f"Request received for POST /{mcp_name}/disable")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for POST /{mcp_name}/disable.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")
    installer = MCPInstaller(db_session=db)
    updated_mcp = installer.disable_mcp(mcp_name=mcp_name)
    logger.info(f"Successfully processed disable request for MCP: {mcp_name}. Current status: {'disabled' if not updated_mcp.is_enabled else 'still enabled'}")
    return updated_mcp


@router.delete("/{mcp_name}", status_code=status.HTTP_204_NO_CONTENT)
async def uninstall_mcp_endpoint(mcp_name: str, db: Session = Depends(get_db_session)):
    """
    Uninstalls an MCP.
    """
    logger.info(f"Request received for DELETE /{mcp_name}")
    if db is None: # Placeholder check
        logger.error(f"Database session not available for DELETE /{mcp_name}.")
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Database session not available")
    installer = MCPInstaller(db_session=db)
    success = installer.uninstall_mcp(mcp_name=mcp_name)
    if success:
        logger.info(f"Successfully processed uninstall request for MCP: {mcp_name}")
    # If uninstall_mcp fails, it raises an exception caught by handlers.
    # If it returns False (though current impl raises), this would be an issue.
    # The current MCPInstaller.uninstall_mcp raises on failure or returns True.
    return None


# --- Instructions for Integration (to be included in final report) ---
# To integrate this router into your main FastAPI application:
#
# 1. In your main FastAPI file (e.g., `main.py`):
#    ```python
#    from fastapi import FastAPI
#    from mcp_service_manager.api_v1 import router as mcp_api_router
#    # Assuming you have a database setup module
#    # from .database import engine, Base # Base for SQLAlchemy models
#
#    # Create database tables (if not using Alembic migrations)
#    # from mcp_service_manager.models import Base as MCPBase
#    # MCPBase.metadata.create_all(bind=engine) # If models are in mcp_service_manager
#
#    app = FastAPI(title="MCP Service Management API")
#
#    app.include_router(mcp_api_router)
#
#    # @app.get("/")
#    # def read_root():
#    #     return {"message": "Welcome to MCP Service Manager"}
#    ```
#
# 2. Database Session Dependency (`get_db_session`):
#    The provided `get_db_session` is a placeholder. You need to replace its
#    content with your actual SQLAlchemy session provider. Typically, this involves:
#    ```python
#    # In a database.py or similar:
#    # from sqlalchemy import create_engine
#    # from sqlalchemy.orm import sessionmaker
#    # from sqlalchemy.ext.declarative import declarative_base
#
#    # SQLALCHEMY_DATABASE_URL = "your_database_url_here" # e.g., "postgresql://user:password@postgresserver/db"
#    # engine = create_engine(SQLALCHEMY_DATABASE_URL)
#    # SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
#    # Base = declarative_base() # Your SQLAlchemy models should inherit from this
#
#    # def get_db():
#    #     db = SessionLocal()
#    #     try:
#    #         yield db
#    #     finally:
#    #         db.close()
#    ```
#    Then, ensure `get_db_session` in `api_v1.py` uses this `get_db`.
#
# 3. Model Creation:
#    Make sure all SQLAlchemy models (like `InstalledMCP`) have their tables
#    created in the database. This can be done via `Base.metadata.create_all(bind=engine)`
#    on app startup or through a migration tool like Alembic.
#
# 4. Configuration for ServiceRegistry:
#    The `ServiceRegistry` might need configuration (e.g., `mcp_so_url`). This
#    can be managed via FastAPI's settings management or by providing arguments
#    during its instantiation, possibly through another dependency.
#
# 5. Testing:
#    Thoroughly test each endpoint with valid and invalid inputs, and ensure
#    error handling works as expected. Mock the database session for unit tests if needed.

# Note: The PUT /{mcp_name}/config endpoint does not update the `last_updated`
# timestamp in the database for the MCP record in this version. This would
# require either a direct DB update call here or a new method in MCPInstaller.
# For simplicity in this iteration, that specific field update is omitted but
# would be a good enhancement.
# Also, the `get_db_session` dependency is a placeholder and will raise
# 501 Not Implemented if used as is, as it yields None. This must be
# replaced with actual DB session logic for the API to function.
# The /available endpoint is an exception as it doesn't rely on the DB.
#
# The `mcp_config.json` is assumed to be located at `mcp_base_dir / mcp_name / MCP_CONFIG_FILE_NAME`
# as per MCPInstaller logic. The PUT endpoint for config update relies on `mcp_record.config_file_path`
# being correctly set by `MCPInstaller.install_mcp()`.
# The `MCPConfig` class itself is not directly used for writing in the PUT endpoint,
# as the requirement was to update the file with the provided JSON body.
# `MCPConfig.load_config()` could be used to validate the structure before writing if needed.
#
# Reviewing the `MCPListItem` and `InstalledMCP` model: ensure all fields match,
# especially `installed_at` and `last_updated` optionality and types. Schemas reflect this.
# `MCPAvailableListItem` is based on assumptions for mcp.so.
# `MCPStatus` is currently an alias for `MCPListItem`.
#
# All custom exceptions from the `mcp_service_manager` are caught and mapped to
# appropriate HTTP status codes. A generic `MCPError` handler is also included.
#
# The code includes instructions for integration in comments at the end.
# Final check on dependencies: FastAPI, Pydantic, SQLAlchemy (for real DB session).
# The `mcp_service_manager` package itself.
# `requests` library is used by `ServiceRegistry` and `MCPInstaller`.
# Standard libraries: `json`, `pathlib`, `typing`.
#
# The solution uses the `APIRouter` from FastAPI and includes all specified endpoints.
# Pydantic models from `schemas.py` are used for request and response validation.
# The database session dependency `get_db_session` is a placeholder that needs
# to be implemented in a real application.
# Error handling for custom exceptions is implemented using `@router.exception_handler`.
# The `PUT /{mcp_name}/config` endpoint writes directly to the file path stored in `InstalledMCP.config_file_path`.
# It does not currently update the `last_updated` field in the database for this action, this is noted as an enhancement.
# The `DELETE /{mcp_name}` endpoint returns HTTP 204 as is conventional.
# The `GET /available` endpoint instantiates `ServiceRegistry` directly; in a larger app, this might be a managed dependency.
# All other endpoints requiring DB access depend on `get_db_session`. If `db is None` (due to placeholder), they raise 501.
# This is a safety measure for the current environment where a live DB is not set up by the tools.
# In a real deployment, `get_db_session` would always provide a session or raise an error itself.The `mcp_service_manager/schemas.py` file was created in the previous turn.
The `mcp_service_manager/api_v1.py` file has now been created.

It includes:
*   An `APIRouter` instance.
*   A placeholder `get_db_session` dependency that needs to be replaced with actual database session logic. For now, it yields `None` and endpoints using it will return HTTP 501 if not mocked, except for `/available`.
*   Exception handlers for all custom exceptions from the `mcp_service_manager` package, mapping them to appropriate HTTP status codes and responses.
*   Implementations for all specified API endpoints:
    *   `GET /available`: Uses `ServiceRegistry`.
    *   `GET /installed`: Uses `MCPInstaller.list_installed_mcps()`.
    *   `POST /install`: Uses `MCPInstaller.install_mcp()`.
    *   `GET /{mcp_name}`: Uses `ServiceMonitor.get_mcp_status()`.
    *   `PUT /{mcp_name}/config`: Reads the `config_file_path` from the `InstalledMCP` record and writes the new JSON configuration directly to this file. *Note: Updating `last_updated` in the database for this operation is currently omitted as an explicit step but noted as a potential enhancement.*
    *   `POST /{mcp_name}/enable`: Uses `MCPInstaller.enable_mcp()`.
    *   `POST /{mcp_name}/disable`: Uses `MCPInstaller.disable_mcp()`.
    *   `DELETE /{mcp_name}`: Uses `MCPInstaller.uninstall_mcp()`, returns HTTP 204.
*   Pydantic models from `schemas.py` are used for request body validation and response serialization.
*   Comments at the end of the file provide instructions on how to integrate the router and the database session dependency into a main FastAPI application.

**Important Considerations from the generated code:**
*   The `get_db_session` is a placeholder. For the API to function with a database, this must be implemented correctly.
*   The `PUT /{mcp_name}/config` endpoint directly modifies the JSON file. It does not currently trigger an update of the `InstalledMCP.last_updated` field in the database. This could be added by calling a specific method on the `installer` or by directly manipulating the `mcp_record` and committing the session.
*   The `/available` endpoint instantiates `ServiceRegistry` directly. In a more complex application, this might be managed as a dependency or singleton with configuration injected.

All core requirements for the API endpoints, Pydantic schemas, and basic integration instructions are covered.
