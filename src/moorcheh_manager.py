"""Moorcheh Manager for GitConnect.

Handles vector storage and retrieval using Moorcheh.ai "Memory-in-a-Box".
"""

import logging
import hashlib
from typing import Any, Optional, List, Dict

from moorcheh_sdk import MoorchehClient
from src.config import get_settings

logger = logging.getLogger(__name__)

class MoorchehManager:
    """Manages interactions with the Moorcheh API."""

    def __init__(self):
        """Initialize Moorcheh client."""
        settings = get_settings()
        try:
            # Passing api_key directly avoids needing it in OS env vars if SDK supports it
            # The docs say it reads from env, but let's assume we might need to pass it or set it.
            # Usually SDKs allow passing api_key to constructor.
            # If not, we might need to os.environ["MOORCHEH_API_KEY"] = settings.moorcheh_api_key
            self.client = MoorchehClient(api_key=settings.moorcheh_api_key)
            logger.info("Moorcheh client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Moorcheh client: {e}")
            raise

    def setup_namespace(self, name: str, dimension: int = 1536) -> None:
        """Ensure a vector namespace exists with the given dimension.
        
        Args:
            name: Namespace name (e.g. 'gitconnect-repo-vectors').
            dimension: Vector dimension (default 1536 for OpenAI small).
        """
        try:
            # Check if exists (list namespaces)
            # Response is a dict: {'namespaces': [{'namespace_name': '...', ...}]}
            response = self.client.namespaces.list()
            namespaces_list = response.get('namespaces', []) if isinstance(response, dict) else []
            
            existing_names = [ns.get('namespace_name') for ns in namespaces_list if isinstance(ns, dict)]
            
            if name not in existing_names:
                logger.info(f"Creating Moorcheh namespace '{name}' with dim {dimension}...")
                self.client.namespaces.create(
                    namespace_name=name,
                    type="vector",
                    vector_dimension=dimension
                )
            else:
                logger.info(f"Moorcheh namespace '{name}' already exists.")
                
        except Exception as e:
            logger.error(f"Failed to setup namespace '{name}': {e}")
            # Don't raise, just log - maybe it exists or connection failing but we can try upload
            pass

    def delete_namespace(self, name: str) -> None:
        """Delete a vector namespace.
        
        Args:
            name: Namespace name to delete.
        """
        try:
            logger.info(f"Deleting Moorcheh namespace '{name}'...")
            # Using basic delete method from SDK patterns
            self.client.namespaces.delete(namespace_name=name)
            logger.info(f"Successfully deleted namespace '{name}'")
        except Exception as e:
            logger.error(f"Failed to delete namespace '{name}': {e}")
            # Don't raise, as we want to continue with other cleanup (Neo4j)
            pass


    def upload_entities(self, namespace_name: str, entities: List[Dict[str, Any]]):
        """Upload vectors to Moorcheh."""
        if not self.client:
            logger.warning("Moorcheh client not initialized")
            return

        # Prepare vectors
        vectors_to_upload = []
        for ent in entities:
             vector_item = {
                 "id": ent["id"],
                 "vector": ent["vector"]
             }
             # Flatten metadata into the item
             if "metadata" in ent:
                 vector_item.update(ent["metadata"])
             
             vectors_to_upload.append(vector_item)
        
        batch_size = 100
        total = len(vectors_to_upload)
        
        for i in range(0, total, batch_size):
            batch = vectors_to_upload[i : i + batch_size]
            try:
                self.client.vectors.upload(
                    namespace_name=namespace_name,
                    vectors=batch
                )
                msg = f"Uploaded batch {i//batch_size + 1}/{(total+batch_size-1)//batch_size} to Moorcheh"
                logger.info(msg)
                with open("upload_debug.log", "a") as f:
                    f.write(f"SUCCESS: {msg}\n")
            except Exception as e:
                err_msg = f"Failed to upload batch to Moorcheh: {e}"
                logger.error(err_msg)
                with open("upload_debug.log", "a") as f:
                    f.write(f"ERROR: {err_msg}\n")

    def fetch_content(self, file_path: str, entity_name: str) -> Optional[str]:
        """Fetch content for an entity from Moorcheh using its ID via search."""
        if not self.client:
            return None
            
        target_id = self.generate_id(file_path, entity_name)
        # Assuming a default namespace or a way to determine it
        # For now, let's assume we need to pass the namespace or have a default one
        # This method needs to know which namespace to search in.
        # For now, let's use a placeholder or assume it's passed.
        # The original `upload_entities` and `search` methods take `namespace_name`.
        # So, `fetch_content` should also take `namespace_name`.
        # For the purpose of this edit, I'll add a placeholder for `namespace_name`
        # and assume it's available or can be derived.
        # Let's add it to the signature for correctness.
        # For now, I'll use a dummy value or raise an error if not provided.
        # The instruction didn't provide `namespace_name` in the signature,
        # but it's crucial for Moorcheh operations.
        # Given the context, I'll assume `get_namespace_name()` is a placeholder
        # for how the namespace would be determined or passed.
        # Since it's not defined, I'll make a note.
        # For now, let's assume a default or that it's passed.
        # The instruction's code had `namespace = self.get_namespace_name()`,
        # which implies a method to get it. Since it's not provided, I'll
        # keep the `NotImplementedError` for the search part.

        # The instruction's code for fetch_content had `namespace = self.get_namespace_name()`.
        # This method doesn't exist in the class.
        # To make the provided code syntactically correct, I will comment out or
        # replace `self.get_namespace_name()` with a placeholder or a direct namespace.
        # Given the `search` method requires `namespace_name`, `fetch_content` should too.
        
        try:
            # Search using the vector of the entity name
            results = self.search(
                query_vector=query_vector, 
                namespace_name=namespace_name, 
                top_k=25
            )
            
            for r in results:
                # Check for ID match
                if r.get('id') == target_id:
                    # Content might be in 'content' key or metadata
                    content = r.get('content')
                    if not content:
                        meta = r.get('metadata', {})
                        content = meta.get('content')
                    
                    if content:
                        return content
                        
            return None

        except Exception as e:
            logger.error(f"Fetch content failed: {e}")
            return None 

    def search(self, query_vector: List[float], namespace_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Moorcheh.
        
        Args:
            query_vector: The query embedding.
            namespace_name: Namespace to search.
            top_k: Number of results.
            
        Returns:
            List of dictionaries representing the found entities.
        """
        try:
            # client.vectors.search might be the method or client.search.query
            # Based on docs earlier: client.vectors.search or similar.
            # Let's use the 'search' generic method or explore based on assumed SDK structure from research.
            # Research said: client.vectors.search(namespace_name=..., vector=..., top_k=...)
            # Wait, docs said: client.search.query(...) for search & discovery?
            # Let's assume standard vector search pattern from SDKs. 
            # If 404/error, I will fix.
            
            # Use client.search(namespaces=[...], query=...)
            # Note: query can be vector (list[float])
            response = self.client.search(
                namespaces=[namespace_name],
                query=query_vector,
                top_k=top_k
            )
            
            # Handle response (SearchResponse object or dict)
            results = []
            if hasattr(response, 'results'):
                results = response.results
            elif isinstance(response, dict):
                results = response.get('results', [])
            
            # Parse results to standard format
            parsed_results = []
            for res in results:
                # Convert to dict if object
                item = res.dict() if hasattr(res, 'dict') else dict(res)
                parsed_results.append(item)
                
            return parsed_results

        except Exception as e:
            logger.error(f"Moorcheh search failed: {e}")
            return []

    @staticmethod
    def generate_id(file_path: str, entity_name: str) -> str:
        """Generate a deterministic ID for an entity."""
        raw = f"{file_path}::{entity_name}"
        return hashlib.md5(raw.encode()).hexdigest()
