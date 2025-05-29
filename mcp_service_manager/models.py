from sqlalchemy import Column, Integer, String, Text, Boolean, TIMESTAMP, func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class InstalledMCP(Base):
    """SQLAlchemy model for the installed_mcps table."""
    __tablename__ = "installed_mcps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    mcp_name = Column(String(255), nullable=False, unique=True)
    mcp_version = Column(String(50))
    source_url = Column(Text)
    local_path = Column(Text, nullable=False)
    config_file_path = Column(Text, nullable=False)
    is_enabled = Column(Boolean, default=True)
    installed_at = Column(TIMESTAMP, default=func.now())
    last_updated = Column(TIMESTAMP, default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<InstalledMCP(id={self.id}, mcp_name='{self.mcp_name}', mcp_version='{self.mcp_version}')>"
