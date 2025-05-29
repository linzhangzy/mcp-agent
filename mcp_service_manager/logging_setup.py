import logging
import logging.handlers
import os

# Default log file paths (can be made configurable)
LOG_DIR = os.path.join(os.getcwd(), "logs") # Create a 'logs' subdirectory
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
MCP_LOG_FILE = os.path.join(LOG_DIR, "mcp_operations.log")

DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
DEFAULT_BACKUP_COUNT = 5

def setup_logging(log_level=logging.INFO, max_bytes=DEFAULT_MAX_BYTES, backup_count=DEFAULT_BACKUP_COUNT):
    """
    Programmatically configures the Python logging system based on project document requirements.
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(LOG_DIR):
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            # Handle error if directory creation fails (e.g., permissions)
            # For now, print to stderr and continue (logging to files might fail)
            print(f"Warning: Could not create log directory {LOG_DIR}. Error: {e}")
            # Fallback to current working directory if LOG_DIR creation fails
            global APP_LOG_FILE, MCP_LOG_FILE
            APP_LOG_FILE = "app.log"
            MCP_LOG_FILE = "mcp_operations.log"


    # 1. Formatters
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 2. Handlers
    # File handler for app.log
    file_handler = logging.handlers.RotatingFileHandler(
        filename=APP_LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(log_level) # Set level for the handler itself

    # File handler for mcp_operations.log
    mcp_file_handler = logging.handlers.RotatingFileHandler(
        filename=MCP_LOG_FILE,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    mcp_file_handler.setFormatter(detailed_formatter)
    mcp_file_handler.setLevel(log_level) # Set level for the handler itself

    # 3. Loggers
    # agent_core logger
    agent_core_logger = logging.getLogger("agent_core")
    agent_core_logger.addHandler(file_handler)
    agent_core_logger.setLevel(log_level)
    agent_core_logger.propagate = False # Prevent passing messages to the root logger if it has handlers

    # mcp_manager logger
    mcp_manager_logger = logging.getLogger("mcp_manager")
    mcp_manager_logger.addHandler(mcp_file_handler)
    mcp_manager_logger.addHandler(file_handler) # Also logs to app.log as per diagram
    mcp_manager_logger.setLevel(log_level)
    mcp_manager_logger.propagate = False

    # 4. Root Logger Configuration (Optional, but good practice)
    # Configure root logger to catch anything not handled by specific loggers,
    # but be careful not to duplicate logs if specific loggers also propagate.
    # For this setup, since specific loggers have propagate=False,
    # we can configure the root logger without duplicating those.
    # However, the spec doesn't explicitly ask for root logger handlers,
    # so we'll just set its level. If other parts of an app use logging.getLogger()
    # without a specific name, they'd use the root.
    root_logger = logging.getLogger()
    if not root_logger.handlers: # Add a default handler if root has no handlers
        # This could be a StreamHandler to console for example,
        # or another file handler if desired.
        # For now, let's ensure it has a level if it's used.
        pass # No specific root handlers defined in the doc, only for agent_core and mcp_manager
    root_logger.setLevel(logging.WARNING) # Default to WARNING for unspecified loggers

    # To test the setup (optional, can be removed)
    # agent_core_logger.info("Logging setup complete for agent_core.")
    # mcp_manager_logger.info("Logging setup complete for mcp_manager.")
    # logging.getLogger("other_module").warning("This is a warning from an other module.")

if __name__ == '__main__':
    # Example of how to use it:
    setup_logging()
    logging.getLogger("agent_core").info("Test agent_core log from main.")
    logging.getLogger("mcp_manager").info("Test mcp_manager log from main.")
    logging.getLogger("mcp_manager").error("Test mcp_manager error log from main.")
    logging.warning("Test root logger warning from main.") # Will go to root if root has handlers
    print(f"Logging to: {APP_LOG_FILE} and {MCP_LOG_FILE}")
