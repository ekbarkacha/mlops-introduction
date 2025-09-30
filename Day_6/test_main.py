from fastapi.testclient import TestClient
from app.main import app,ml_models
from unittest.mock import patch, MagicMock

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.main.load_model", return_value=MagicMock(predict=MagicMock(return_value=[1])))
def test_list_models(mock_load_model):
    from app.main import app, ml_models
    with TestClient(app) as client:
        response = client.get("/models")
        assert response.status_code == 200
        assert "logistic_model" in response.json()["available_models"]
        assert "rf_model" in response.json()["available_models"]


def test_predict_invalid_model():
    response = client.post("/predict/rf_mode", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    # When we use Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")]
    assert response.status_code == 422

    # assert response.status_code == 404 
    # assert response.json() == {"detail": "Model not found"}

@patch("app.main.ml_models", {"logistic_model": MagicMock(predict=MagicMock(return_value=[1]))})
def test_predict_valid_model():
    response = client.post("/predict/logistic_model", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 200
    assert response.json() == {"model": "logistic_model", "prediction": 1}

def test_predict_secure_invalid_model():
    response = client.post("/predict_secure/rf_mode", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    },headers={"X-API-Key": "mysecureapikey123"})
    # When we use Annotated[str, Path(pattern=r"^(logistic_model|rf_model)$")]
    assert response.status_code == 422

    # assert response.status_code == 404 
    # assert response.json() == {"detail": "Model not found"}

@patch("app.main.load_model", return_value=MagicMock(predict=MagicMock(return_value=[1])))
def test_predict_secure_valid_api_key(mock_load_model):
    from app.main import app, ml_models
    with TestClient(app) as client:
        response = client.post("/predict_secure/logistic_model", 
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2}, 
        headers={"X-API-Key": "mysecureapikey123"})

        assert response.status_code == 200
        assert response.json() == {
            "model": "logistic_model",
            "prediction": 1
            }

@patch("app.main.load_model", return_value=MagicMock(predict=MagicMock(return_value=[1])))
def test_predict_secure_missing_api_key(mock_load_model):
    response = client.post("/predict_secure/logistic_model", json={
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    })
    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}



    