
from fastapi.testclient import TestClient
from src.main import app
from src.config import get_settings

client = TestClient(app)
settings = get_settings()

def test_ingest_invalid_url():
    """Test that ingestion rejects non-GitHub URLs."""
    with TestClient(app) as client:
        response = client.post("/ingest", json={
            "repo_url": "https://gitlab.com/user/repo",
            "clear_existing": False
        })
        assert response.status_code == 422
        assert "Must be a valid GitHub URL" in response.text

def test_query_no_auth():
    """Test that query endpoint rejects requests without admin secret."""
    with TestClient(app) as client:
        response = client.post("/query?query=MATCH (n) RETURN n")
        assert response.status_code == 403
        assert "Invalid admin secret" in response.json()["detail"]

def test_query_valid_auth():
    """Test that query endpoint rejects requests with wrong secret."""
    with TestClient(app) as client:
        response = client.post(
            "/query?query=MATCH (n) RETURN n",
            headers={"X-Admin-Secret": "wrong-secret"}
        )
        assert response.status_code == 403

def test_query_valid_auth_success():
    """Test that query endpoint accepts requests with correct secret."""
    with TestClient(app) as client:
        # Mock the execute_cypher method to avoid DB hit
        if hasattr(app.state, "graph_manager"):
            app.state.graph_manager.execute_cypher = lambda q, params=None: [{"result": "success"}]
        
        response = client.post(
            "/query?query=MATCH (n) RETURN n",
            headers={"X-Admin-Secret": settings.admin_secret}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
