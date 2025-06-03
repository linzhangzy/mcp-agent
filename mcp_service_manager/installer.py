import requests
import zipfile
import tarfile
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Any # Any for db_session type hint
from sqlalchemy.orm import Session # For type hinting db_session
from sqlalchemy.exc import SQLAlchemyError
import logging

from .models import InstalledMCP
from .exceptions import (
    MCPDownloadError,
    MCPExtractionError,
    MCPAlreadyInstalledError,
    MCPInstallationError,
    MCPNotFoundError
)

logger = logging.getLogger("mcp_manager")

class MCPInstaller:
    """
    Handles downloading, extracting, and locally integrating MCP packages,
    including database interactions.
    """
    DEFAULT_BASE_INSTALL_PATH = "/app/installed_mcps/"
    MCP_CONTENT_DIR_NAME = "mcp_content"
    MCP_CONFIG_FILE_NAME = "mcp_config.json"

    def __init__(self, db_session: Session, base_install_path: str = DEFAULT_BASE_INSTALL_PATH):
        """
        Initializes the MCPInstaller.

        Args:
            db_session: SQLAlchemy session for database operations.
            base_install_path: The root directory where MCPs will be installed.
        """
        self.db_session = db_session
        self.base_install_path = Path(base_install_path)
        self.base_install_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"MCPInstaller initialized with base_install_path: {self.base_install_path}")

    def _download_package(self, download_url: str, target_dir: Path, mcp_name: str) -> Path:
        """
        Downloads an MCP package.

        Args:
            download_url: URL to download the package from.
            target_dir: Directory to save the downloaded file.
            mcp_name: Name of the MCP, used for naming the downloaded file.

        Returns:
            Path to the downloaded package file.

        Raises:
            MCPDownloadError: If any download-related error occurs.
        """
        target_dir.mkdir(parents=True, exist_ok=True)
        # Determine file extension for naming, default to .package if not obvious
        file_ext = Path(download_url).suffix
        if not file_ext or len(file_ext) > 10: # Basic sanity check for extension
            if ".zip" in download_url.lower():
                 file_ext = ".zip"
            elif ".tar.gz" in download_url.lower():
                 file_ext = ".tar.gz"
            else:
                 file_ext = ".package" # Generic extension

        download_file_name = f"{mcp_name}_package{file_ext}"
        package_path = target_dir / download_file_name

        logger.info(f"Starting download for MCP '{mcp_name}' from {download_url} to {package_path}")
        try:
            with requests.get(download_url, stream=True, timeout=30) as r:
                r.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                with open(package_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            logger.info(f"Successfully downloaded MCP '{mcp_name}' to {package_path}")
            return package_path
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error {e.response.status_code} downloading {download_url} for MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPDownloadError(url=download_url, details=f"HTTP error {e.response.status_code}: {e}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout downloading {download_url} for MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPDownloadError(url=download_url, details="Request timed out.")
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error downloading {download_url} for MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPDownloadError(url=download_url, details="Connection error. Ensure the server is reachable.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception downloading {download_url} for MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPDownloadError(url=download_url, details=f"An unexpected error occurred during download: {e}")
        except IOError as e:
            logger.error(f"IOError writing downloaded file {package_path} for MCP '{mcp_name}': {e}", exc_info=True)
            if package_path.exists():
                os.remove(package_path)
            raise MCPDownloadError(url=download_url, details=f"Failed to write downloaded file: {e}")


    def _extract_package(self, package_path: Path, extract_dir: Path):
        """
        Extracts an MCP package. Supports .zip and .tar.gz.

        Args:
            package_path: Path to the downloaded package file.
            extract_dir: Directory where the package contents should be extracted.

        Raises:
            MCPExtractionError: If extraction fails or format is unsupported.
        """
        logger.info(f"Starting extraction of '{package_path.name}' to '{extract_dir}'")
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            if package_path.name.endswith(".zip"):
                with zipfile.ZipFile(package_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif package_path.name.endswith(".tar.gz"):
                with tarfile.open(package_path, "r:gz") as tar_ref:
                    tar_ref.extractall(extract_dir)
            elif package_path.name.endswith(".tar"): # Also support .tar
                 with tarfile.open(package_path, "r:") as tar_ref:
                    tar_ref.extractall(extract_dir)
            else:
                logger.error(f"Unsupported package format for {package_path.name}.")
                raise MCPExtractionError(
                    package_path=str(package_path),
                    details=f"Unsupported package format: {package_path.name}. Only .zip and .tar.gz/.tar are supported."
                )
            logger.info(f"Successfully extracted '{package_path.name}' to '{extract_dir}'.")
            # Clean up the downloaded archive after successful extraction
            os.remove(package_path)
            logger.debug(f"Removed archive file '{package_path}' after extraction.")
        except (zipfile.BadZipFile, tarfile.ReadError, tarfile.CompressionError) as e:
            logger.error(f"Extraction failed for '{package_path.name}': Corrupted or invalid archive. Details: {e}", exc_info=True)
            raise MCPExtractionError(package_path=str(package_path), details=f"Corrupted or invalid archive: {e}")
        except Exception as e: # Catch-all for other OS or permission errors during extraction
            logger.error(f"Extraction failed for '{package_path.name}': Unexpected error. Details: {e}", exc_info=True)
            raise MCPExtractionError(package_path=str(package_path), details=f"An unexpected error occurred during extraction: {e}")

    def install_mcp(self, mcp_name: str, version: str, source_url: str, download_url: str,
                      config_template: Optional[dict] = None) -> InstalledMCP:
        """
        Orchestrates the installation of an MCP: download, extract, configure, and record in DB.

        Args:
            mcp_name: Name of the MCP.
            version: Version of the MCP.
            source_url: Original source URL of the MCP (e.g., Git repo).
            download_url: Direct download URL for the MCP package.
            config_template: Optional dictionary to populate the initial mcp_config.json.

        Returns:
            The created InstalledMCP object.

        Raises:
            MCPAlreadyInstalledError: If an MCP with the same name already exists.
            MCPDownloadError: If downloading fails.
            MCPExtractionError: If extraction fails.
            MCPInstallationError: For other installation-related issues (e.g., DB errors).
        """
        logger.info(f"Attempting to install MCP: {mcp_name} version: {version} from source: {source_url}, download: {download_url}")
        # 1. Check for existing MCP
        existing_mcp = self.db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).first()
        if existing_mcp:
            logger.warning(f"Attempt to install MCP '{mcp_name}' which is already installed.")
            raise MCPAlreadyInstalledError(mcp_name=mcp_name)

        # 2. Create directory structure
        logger.debug(f"Creating directory structure for MCP '{mcp_name}' at {self.base_install_path}")
        mcp_base_dir = self.base_install_path / mcp_name
        mcp_version_dir = mcp_base_dir / version
        mcp_content_install_dir = mcp_version_dir / self.MCP_CONTENT_DIR_NAME
        temp_download_dir = mcp_base_dir / "temp_download"

        try:
            mcp_base_dir.mkdir(parents=True, exist_ok=True)
            mcp_version_dir.mkdir(parents=True, exist_ok=True)
            mcp_content_install_dir.mkdir(parents=True, exist_ok=True)
            temp_download_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory structure created for MCP '{mcp_name}'.")

            # 3. Download package
            downloaded_package_path = self._download_package(download_url, temp_download_dir, mcp_name)

            # 4. Extract package
            self._extract_package(downloaded_package_path, mcp_content_install_dir)

            # 5. Create mcp_config.json
            config_file_path = mcp_base_dir / self.MCP_CONFIG_FILE_NAME
            config_data = config_template if config_template is not None else {}
            logger.info(f"Creating config file for MCP '{mcp_name}' at {config_file_path}")
            with open(config_file_path, 'w') as f:
                json.dump(config_data, f, indent=4)
            logger.debug(f"Config file created for MCP '{mcp_name}'.")

            # 6. Create InstalledMCP record
            logger.info(f"Adding MCP '{mcp_name}' record to database.")
            new_mcp_record = InstalledMCP(
                mcp_name=mcp_name,
                mcp_version=version,
                source_url=source_url,
                local_path=str(mcp_content_install_dir.resolve()),
                config_file_path=str(config_file_path.resolve()),
                is_enabled=True # Default to enabled
            )
            self.db_session.add(new_mcp_record)
            self.db_session.commit()
            self.db_session.refresh(new_mcp_record)
            logger.info(f"MCP '{mcp_name}' (ID: {new_mcp_record.id}) successfully installed and recorded in database.")
            return new_mcp_record

        except (MCPDownloadError, MCPExtractionError, MCPAlreadyInstalledError) as e:
            logger.error(f"Installation failed for MCP '{mcp_name}': {e}", exc_info=True)
            # Clean up created directories if error occurs mid-process
            if mcp_version_dir.exists():
                logger.debug(f"Cleaning up directories for failed installation of MCP '{mcp_name}' at {mcp_version_dir}")
                shutil.rmtree(mcp_version_dir, ignore_errors=True)
                if mcp_base_dir.exists() and not any(mcp_base_dir.iterdir()):
                     shutil.rmtree(mcp_base_dir, ignore_errors=True)
            self.db_session.rollback()
            raise e
        except SQLAlchemyError as e:
            logger.error(f"Database error during installation of MCP '{mcp_name}': {e}", exc_info=True)
            self.db_session.rollback()
            if mcp_version_dir.exists():
                logger.debug(f"Cleaning up directories for failed installation (DB error) of MCP '{mcp_name}' at {mcp_version_dir}")
                shutil.rmtree(mcp_version_dir, ignore_errors=True)
                if mcp_base_dir.exists() and not any(mcp_base_dir.iterdir()):
                     shutil.rmtree(mcp_base_dir, ignore_errors=True)
            raise MCPInstallationError(f"Database error during MCP installation: {e}")
        except Exception as e: # Catch-all for other unexpected errors
            logger.error(f"Unexpected error during installation of MCP '{mcp_name}': {e}", exc_info=True)
            self.db_session.rollback()
            if mcp_version_dir.exists():
                logger.debug(f"Cleaning up directories for failed installation (unexpected error) of MCP '{mcp_name}' at {mcp_version_dir}")
                shutil.rmtree(mcp_version_dir, ignore_errors=True)
                if mcp_base_dir.exists() and not any(mcp_base_dir.iterdir()):
                     shutil.rmtree(mcp_base_dir, ignore_errors=True)
            raise MCPInstallationError(f"An unexpected error occurred during MCP installation: {e}")
        finally:
            if temp_download_dir.exists():
                logger.debug(f"Cleaning up temporary download directory {temp_download_dir}")
                shutil.rmtree(temp_download_dir, ignore_errors=True)

    def enable_mcp(self, mcp_name: str) -> InstalledMCP:
        """
        Enables an installed MCP.

        Args:
            mcp_name: The name of the MCP to enable.

        Returns:
            The updated InstalledMCP object.

        Raises:
            MCPNotFoundError: If the MCP is not found.
            MCPInstallationError: For database errors.
        """
        logger.info(f"Attempting to enable MCP '{mcp_name}'.")
        mcp_record = self.get_installed_mcp(mcp_name)
        if not mcp_record:
            logger.warning(f"MCP '{mcp_name}' not found for enabling.")
            raise MCPNotFoundError(mcp_name=mcp_name)

        if mcp_record.is_enabled:
            logger.info(f"MCP '{mcp_name}' is already enabled.")
            return mcp_record

        mcp_record.is_enabled = True
        try:
            self.db_session.commit()
            self.db_session.refresh(mcp_record)
            logger.info(f"MCP '{mcp_name}' (ID: {mcp_record.id}) successfully enabled.")
            return mcp_record
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error while enabling MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPInstallationError(f"Database error while enabling MCP '{mcp_name}': {e}")

    def disable_mcp(self, mcp_name: str) -> InstalledMCP:
        """
        Disables an installed MCP.

        Args:
            mcp_name: The name of the MCP to disable.

        Returns:
            The updated InstalledMCP object.

        Raises:
            MCPNotFoundError: If the MCP is not found.
            MCPInstallationError: For database errors.
        """
        logger.info(f"Attempting to disable MCP '{mcp_name}'.")
        mcp_record = self.get_installed_mcp(mcp_name)
        if not mcp_record:
            logger.warning(f"MCP '{mcp_name}' not found for disabling.")
            raise MCPNotFoundError(mcp_name=mcp_name)

        if not mcp_record.is_enabled:
            logger.info(f"MCP '{mcp_name}' is already disabled.")
            return mcp_record

        mcp_record.is_enabled = False
        try:
            self.db_session.commit()
            self.db_session.refresh(mcp_record)
            logger.info(f"MCP '{mcp_name}' (ID: {mcp_record.id}) successfully disabled.")
            return mcp_record
        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error while disabling MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPInstallationError(f"Database error while disabling MCP '{mcp_name}': {e}")

    def uninstall_mcp(self, mcp_name: str) -> bool:
        """
        Removes an MCP's files and its database record.
        This includes the MCP's base installation directory and its configuration file.

        Args:
            mcp_name: The name of the MCP to uninstall.

        Returns:
            True if uninstallation was successful.

        Raises:
            MCPNotFoundError: If the MCP is not found.
            MCPInstallationError: If a DB or filesystem error occurs.
        """
        logger.info(f"Attempting to uninstall MCP '{mcp_name}'.")
        mcp_record = self.get_installed_mcp(mcp_name)
        if not mcp_record:
            logger.warning(f"MCP '{mcp_name}' not found for uninstallation.")
            raise MCPNotFoundError(mcp_name=mcp_name)

        mcp_id_for_logging = mcp_record.id # Get ID before record is deleted
        mcp_base_dir_to_delete = self.base_install_path / mcp_name
        logger.debug(f"Identified base directory for uninstallation of MCP '{mcp_name}': {mcp_base_dir_to_delete}")

        try:
            if mcp_base_dir_to_delete.exists():
                if not mcp_base_dir_to_delete.is_dir():
                    logger.error(f"Path {mcp_base_dir_to_delete} for MCP '{mcp_name}' is not a directory.")
                    raise MCPInstallationError(
                        f"Expected a directory but found a file at {mcp_base_dir_to_delete} for MCP '{mcp_name}'."
                    )
                logger.info(f"Deleting directory {mcp_base_dir_to_delete} for MCP '{mcp_name}'.")
                shutil.rmtree(mcp_base_dir_to_delete)
                logger.debug(f"Directory {mcp_base_dir_to_delete} deleted for MCP '{mcp_name}'.")
            else:
                logger.warning(f"Installation directory {mcp_base_dir_to_delete} for MCP '{mcp_name}' not found. Skipping filesystem delete.")

            logger.info(f"Deleting database record for MCP '{mcp_name}' (ID: {mcp_id_for_logging}).")
            self.db_session.delete(mcp_record)
            self.db_session.commit()
            logger.info(f"MCP '{mcp_name}' (ID: {mcp_id_for_logging}) successfully uninstalled.")
            return True

        except SQLAlchemyError as e:
            self.db_session.rollback()
            logger.error(f"Database error during uninstallation of MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPInstallationError(f"Database error during MCP uninstallation for '{mcp_name}': {e}")
        except OSError as e:
            self.db_session.rollback()
            logger.error(f"Filesystem error during uninstallation of MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPInstallationError(f"Filesystem error during MCP uninstallation for '{mcp_name}': {e}")
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Unexpected error during uninstallation of MCP '{mcp_name}': {e}", exc_info=True)
            raise MCPInstallationError(f"An unexpected error occurred during uninstallation of '{mcp_name}': {e}")

    def get_installed_mcp(self, mcp_name: str) -> Optional[InstalledMCP]:
        """
        Retrieves an installed MCP record from the database.

        Args:
            mcp_name: The name of the MCP to retrieve.

        Returns:
            The InstalledMCP object if found, else None.
        """
        logger.debug(f"Querying database for installed MCP with name: {mcp_name}")
        mcp = self.db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).first()
        if mcp:
            logger.debug(f"Found MCP '{mcp_name}' in database (ID: {mcp.id}).")
        else:
            logger.debug(f"MCP '{mcp_name}' not found in database.")
        return mcp

    def list_installed_mcps(self) -> list[InstalledMCP]:
        """
        Lists all installed MCPs from the database.

        Returns:
            A list of InstalledMCP objects.
        """
        logger.debug("Querying database for all installed MCPs.")
        mcps = self.db_session.query(InstalledMCP).all()
        logger.info(f"Retrieved {len(mcps)} installed MCP(s) from the database.")
        return mcps
