from sqlalchemy.orm import Session
from typing import List, Dict, Any
import logging

from .models import InstalledMCP
from .exceptions import MCPNotFoundError

logger = logging.getLogger("mcp_manager")

class ServiceMonitor:
    """
    Provides methods to monitor the status and health of installed MCP services.
    """
    def __init__(self, db_session: Session):
        """
        Initializes the ServiceMonitor.

        Args:
            db_session: SQLAlchemy session for database operations.
        """
        self.db_session = db_session
        logger.debug("ServiceMonitor initialized.")

    def get_mcp_status(self, mcp_name: str) -> Dict[str, Any]:
        """
        Retrieves the status of a specific MCP.

        Args:
            mcp_name: The name of the MCP to get status for.

        Returns:
            A dictionary containing status information.
            Example: {"name": "MyMCP", "is_enabled": True, "version": "1.0",
                      "installed_at": "2023-01-01T12:00:00", "last_updated": "2023-01-01T12:00:00"}

        Raises:
            MCPNotFoundError: If the MCP is not found.
        """
        logger.info(f"Attempting to get status for MCP: {mcp_name}")
        mcp = self.db_session.query(InstalledMCP).filter_by(mcp_name=mcp_name).first()
        if not mcp:
            logger.warning(f"MCP '{mcp_name}' not found in database for status retrieval.")
            raise MCPNotFoundError(mcp_name=mcp_name)

        logger.info(f"Successfully retrieved status for MCP: {mcp_name} (ID: {mcp.id})")
        return {
            "name": mcp.mcp_name,
            "version": mcp.mcp_version,
            "is_enabled": mcp.is_enabled,
            "installed_at": mcp.installed_at.isoformat() if mcp.installed_at else None,
            "last_updated": mcp.last_updated.isoformat() if mcp.last_updated else None,
            "local_path": mcp.local_path,
            "config_file_path": mcp.config_file_path
        }

    def list_all_mcp_statuses(self) -> List[Dict[str, Any]]:
        """
        Retrieves the status of all installed MCPs.

        Returns:
            A list of dictionaries, where each dictionary contains status information for an MCP.
        """
        logger.info("Attempting to list statuses for all installed MCPs.")
        mcps = self.db_session.query(InstalledMCP).all()
        statuses = []
        for mcp in mcps:
            logger.debug(f"Compiling status for MCP: {mcp.mcp_name} (ID: {mcp.id})")
            statuses.append({
                "name": mcp.mcp_name,
                "version": mcp.mcp_version,
                "is_enabled": mcp.is_enabled,
                "installed_at": mcp.installed_at.isoformat() if mcp.installed_at else None,
                "last_updated": mcp.last_updated.isoformat() if mcp.last_updated else None,
                "local_path": mcp.local_path,
                "config_file_path": mcp.config_file_path
            })
        logger.info(f"Successfully compiled statuses for {len(statuses)} MCPs.")
        return statuses
