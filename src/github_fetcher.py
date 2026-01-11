"""GitHub repository fetcher for GitConnect.

Handles cloning repositories from GitHub for parsing.
Supports both public and private repositories (with token).
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from git import Repo, GitCommandError

from src.config import get_settings

logger = logging.getLogger(__name__)


class GitHubFetchError(Exception):
    """Raised when repository fetching fails."""
    pass


class GitHubFetcher:
    """Fetches repositories from GitHub for analysis."""

    SUPPORTED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs"}

    def __init__(self, github_token: Optional[str] = None):
        """Initialize the fetcher.
        
        Args:
            github_token: Optional GitHub token for private repositories.
        """
        self.github_token = github_token or get_settings().github_token
        self._temp_dir: Optional[Path] = None

    def _parse_github_url(self, repo_url: str) -> tuple[str, str]:
        """Parse GitHub URL to extract owner and repo name.
        
        Args:
            repo_url: GitHub repository URL.
            
        Returns:
            Tuple of (owner, repo_name).
            
        Raises:
            GitHubFetchError: If URL is invalid.
        """
        parsed = urlparse(repo_url)
        
        # Handle various URL formats
        path = parsed.path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]
        
        parts = path.split("/")
        if len(parts) < 2:
            raise GitHubFetchError(f"Invalid GitHub URL: {repo_url}")
        
        return parts[0], parts[1]

    def _get_authenticated_url(self, repo_url: str) -> str:
        """Add authentication to GitHub URL if token is available.
        
        Args:
            repo_url: Original repository URL.
            
        Returns:
            URL with authentication if token is available.
        """
        if not self.github_token:
            return repo_url
        
        parsed = urlparse(repo_url)
        if parsed.scheme == "https" and "github.com" in parsed.netloc:
            # Format: https://<token>@github.com/owner/repo.git
            return f"https://{self.github_token}@github.com{parsed.path}"
        
        return repo_url

    def clone_repository(
        self,
        repo_url: str,
        target_dir: Optional[str] = None,
        shallow: bool = True,
    ) -> Path:
        """Clone a GitHub repository.
        
        Args:
            repo_url: GitHub repository URL.
            target_dir: Optional target directory. If None, uses temp directory.
            shallow: If True, performs a shallow clone (depth=1).
            
        Returns:
            Path to the cloned repository.
            
        Raises:
            GitHubFetchError: If cloning fails.
        """
        owner, repo_name = self._parse_github_url(repo_url)
        
        # Determine target directory  
        if target_dir:
            clone_path = Path(target_dir) / repo_name
        else:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="gitconnect_"))
            clone_path = self._temp_dir / repo_name
        
        # Ensure parent directory exists
        clone_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get authenticated URL
        auth_url = self._get_authenticated_url(repo_url)
        
        logger.info(f"Cloning repository {owner}/{repo_name} to {clone_path}")
        
        try:
            # Clone with optional shallow depth and single branch for speed
            clone_kwargs = {"depth": 1, "single_branch": True} if shallow else {}
            
            if shallow:
                logger.info(f"Shallow cloning {owner}/{repo_name} (depth=1, single branch)...")
            else:
                logger.info(f"Cloning full repository {owner}/{repo_name}...")

            Repo.clone_from(auth_url, clone_path, **clone_kwargs)
            logger.info(f"Successfully cloned {owner}/{repo_name}")
            return clone_path
            
        except GitCommandError as e:
            raise GitHubFetchError(f"Failed to clone repository: {e}") from e

    def get_source_files(self, repo_path: Path) -> list[Path]:
        """Get all supported source files from repository.
        
        Args:
            repo_path: Path to the cloned repository.
            
        Returns:
            List of paths to source files.
        """
        source_files = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-source directories
            dirs[:] = [
                d for d in dirs 
                if not d.startswith(".") 
                and d not in {"node_modules", "__pycache__", "venv", "env", "dist", "build"}
            ]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    source_files.append(file_path)
        
        logger.info(f"Found {len(source_files)} source files")
        return source_files

    def cleanup(self) -> None:
        """Clean up temporary directory if created."""
        if self._temp_dir and self._temp_dir.exists():
            logger.info(f"Cleaning up temporary directory: {self._temp_dir}")
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def __enter__(self) -> "GitHubFetcher":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.cleanup()


def fetch_repository(repo_url: str, target_dir: Optional[str] = None) -> tuple[Path, list[Path]]:
    """Convenience function to fetch and list source files from a repository.
    
    Args:
        repo_url: GitHub repository URL.
        target_dir: Optional target directory.
        
    Returns:
        Tuple of (repo_path, source_files).
    """
    fetcher = GitHubFetcher()
    repo_path = fetcher.clone_repository(repo_url, target_dir)
    source_files = fetcher.get_source_files(repo_path)
    return repo_path, source_files
