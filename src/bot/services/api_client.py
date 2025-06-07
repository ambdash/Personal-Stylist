import httpx
import os
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.timeout = 30.0
    
    async def search_nodes_by_word(self, word: str, limit: int = 10) -> Dict[str, Any]:
        """Search for nodes containing the word via API"""
        try:
            logger.info(f"Searching for word: '{word}' via API at {self.base_url}")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"{self.base_url}/telegram/db/search"
                payload = {"word": word, "limit": limit}
                logger.info(f"Making POST request to: {url} with payload: {payload}")
                
                response = await client.post(url, json=payload)
                logger.info(f"Response status: {response.status_code}")
                
                response.raise_for_status()
                result = response.json()
                logger.info(f"API response: found={result.get('found', False)}, nodes_count={len(result.get('nodes', []))}")
                return result
        except Exception as e:
            logger.error(f"Error searching nodes via API: {e}")
            logger.error(f"API base URL: {self.base_url}")
            return {"found": False, "error": str(e)}
    
    async def add_node(self, name: str, node_type: str) -> Dict[str, Any]:
        """Add a new node via API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/telegram/db/nodes",
                    json={"name": name, "node_type": node_type}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error adding node via API: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_node(self, node_id: str) -> Dict[str, Any]:
        """Delete a node via API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.delete(
                    f"{self.base_url}/telegram/db/nodes/{node_id}"
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error deleting node via API: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_relationship(
        self, 
        from_node_id: str, 
        to_node_id: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a relationship via API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/telegram/db/relationships",
                    json={
                        "from_node_id": from_node_id,
                        "to_node_id": to_node_id,
                        "relationship_type": relationship_type,
                        "properties": properties
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error creating relationship via API: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_nodes_by_type(self, node_type: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Get nodes by type via API"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/telegram/db/nodes/by-type/{node_type}",
                    params={"limit": limit}
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error getting nodes by type via API: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check via API"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/telegram/db/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Health check failed via API: {e}")
            return {"status": "unhealthy", "error": str(e)}

# Create a singleton instance
api_client = APIClient() 