from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# For orm_mode compatibility
class OrmBaseModel(BaseModel):
    class Config:
        orm_mode = True

# --- Request Models ---

class MCPInstallRequest(BaseModel):
    mcp_name: str = Field(..., example="MyCoolMCP")
    version: str = Field(..., example="1.0.0")
    source_url: str = Field(..., example="https://github.com/user/mycoolmcp.git")
    download_url: str = Field(..., example="https://github.com/user/mycoolmcp/archive/v1.0.0.zip")
    config_template: Optional[Dict[str, Any]] = Field(None, example={"param1": "value1", "port": 8080})

class MCPConfigUpdateRequest(BaseModel):
    config_json: Dict[str, Any] = Field(..., example={"param1": "new_value", "port": 8081, "enabled": True})

# --- Response Models ---

class MCPListItem(OrmBaseModel):
    id: int
    mcp_name: str
    mcp_version: Optional[str] = None
    source_url: Optional[str] = None
    local_path: str
    config_file_path: str
    is_enabled: bool
    installed_at: Optional[datetime] = None # Made optional as it might not always be set if record is old
    last_updated: Optional[datetime] = None # Made optional

class MCPAvailableListItem(BaseModel):
    # Assuming fields from a typical service discovery like mcp.so
    # These might differ based on actual mcp.so API response structure
    id: str = Field(..., example="my-cool-mcp") # A unique identifier for the service
    name: str = Field(..., example="My Cool MCP")
    description: Optional[str] = Field(None, example="A very cool MCP for doing things.")
    version: Optional[str] = Field(None, example="1.0.1") # Latest available version
    download_url: Optional[str] = Field(None, example="https://mcp.so/download/my-cool-mcp/1.0.1.zip")
    # Other fields like 'author', 'tags' could be added if mcp.so provides them

class MCPStatus(MCPListItem): # For now, status includes all details from MCPListItem
    # Can be a subset or include more runtime-specific status fields in the future
    pass

class GeneralResponse(BaseModel):
    message: str
    detail: Optional[str] = None

class ErrorDetail(BaseModel):
    detail: str

# Example of a more specific error model if needed
class MCPNotFoundErrorDetail(BaseModel):
    mcp_name: str
    message: str
