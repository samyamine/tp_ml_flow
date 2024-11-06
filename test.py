import requests
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test the /predict endpoint
predict_response = client.post(
    "/predict",
    json={"features": [[5.1, 3.5, 1.4, 0.2]]}
)
print("Predict response:", predict_response.json())

# Test the /update-model endpoint
update_response = client.post("/update-model")
print("Update response:", update_response.json())
