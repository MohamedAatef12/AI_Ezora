import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommendation_success():
    payload = {
        "mood": "happy",
        "occasion": "work",
        "preferred_categories": ["scarves", "heels"]
    }

    response = client.post("/api/v1/recommendation/recommend", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "recommended_products" in data
    assert isinstance(data["recommended_products"], list)

    if data["recommended_products"]:
        product = data["recommended_products"][0]
        assert "product_id" in product
        assert "product_type" in product
        assert "style" in product
        assert "color" in product
        assert "season" in product

def test_recommendation_invalid_mood():
    payload = {
        "mood": "invalid_mood",
        "occasion": "work",
        "preferred_categories": ["scarves", "heels"]
    }

    response = client.post("/api/v1/recommendation/recommend", json=payload)

    assert response.status_code == 400
    assert "Unknown mood" in response.json()["detail"]



def test_recommendation_no_results():
    payload = {
        "mood": "happy",
        "occasion": "work",
        "preferred_categories": ["hats"]  # assume it's not encoded
    }

    response = client.post("/api/v1/recommendation/recommend", json=payload)

    assert response.status_code == 200
    assert response.json()["recommended_products"] == []
