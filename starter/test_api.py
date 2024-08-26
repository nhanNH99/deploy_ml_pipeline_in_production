import pytest
from fastapi.testclient import TestClient

from main import ai  # Replace with the actual module name

client = TestClient(ai)


def test_get_info():
    """
    Test the root endpoint to ensure it returns the correct welcome message.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the MLOps project of Tran Duy Nhat Anh"
    }


def test_inference_success():
    """
    Test the /inference endpoint to ensure it returns the correct response when given valid input data.
    """
    input_data = {
        "age": 45,
        "capital_gain": 2174,
        "capital_loss": 0,
        "education": "Bachelors",
        "education_num": 13,
        "fnlwgt": 2334,
        "hours_per_week": 60,
        "marital_status": "Never-married",
        "native_country": "Cuba",
        "occupation": "Prof-specialty",
        "race": "Black",
        "relationship": "Wife",
        "sex": "Female",
        "workclass": "State-gov",
    }

    response = client.post("/inference", json=input_data)
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["message"] == "Salary <= 50K"


def test_inference_invalid_data():
    """
    Test the /inference endpoint to check the missing input data.
    """
    input_data = {
        "age": 45,
        "capital_gain": 2174,
        "capital_loss": 0,
        "education": "Bachelors",
        "education_num": 13,
        "fnlwgt": 2334,
        "hours_per_week": 60,
        "race": "Black",
        "relationship": "Wife",
        "sex": "Female",
        "workclass": "State-gov",
    }

    response = client.post("/inference", json=input_data)
    assert response.status_code == 422
    assert "detail" in response.json()
