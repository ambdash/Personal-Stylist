from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class NodeCreate(BaseModel):
    """Schema for creating a new node"""
    name: str
    label: str
    aliases: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None

class NodeUpdate(BaseModel):
    """Schema for updating a node"""
    name: Optional[str] = None
    aliases: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None

class RelationCreate(BaseModel):
    """Schema for creating a new relationship"""
    start_node: str
    end_node: str
    rel_type: str
    properties: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    """Schema for search queries"""
    query: str
    label: Optional[str] = None
    limit: Optional[int] = 10 