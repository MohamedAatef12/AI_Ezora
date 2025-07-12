import sys
import os

# أضف المسار الجذر للمشروع إلى sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app 
client = TestClient(app)

def test_recommendation_endpoint_success():
    payload = {
        "mood": "romantic",
        "occasion": "party",
        "preferred_categories": ["heels", "skirts"]
    }

    response = client.post("/api/v1/recommendation/recommend", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "recommended_products" in data
    assert isinstance(data["recommended_products"], list)
    assert len(data["recommended_products"]) > 0

    first_product = data["recommended_products"][0]
    assert "product_id" in first_product
    assert "product_type" in first_product
    assert "style" in first_product
    assert "color" in first_product
    assert "season" in first_product
    assert "interaction" in first_product
    assert "timestamp" in first_product
