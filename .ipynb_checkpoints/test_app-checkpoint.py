import pytest
import json
import requests
from app import app  
from pytest_mock import mocker
import shap

@pytest.fixture
def heroku_url():
    """Return the deployed Heroku app URL."""
    return "https://still-bayou-61593-4aed81ce9738.herokuapp.com"


def test_home_endpoint(heroku_url):
    """Test the home endpoint of the deployed Heroku app."""
    response = requests.get(f"{heroku_url}/")
    
    assert response.status_code == 200
    assert 'Hello, this is my Dashboard home page!' in response.text
    print("Response content:", response.text)

def test_predict_endpoint(heroku_url):
    """Test the predict endpoint of the deployed Heroku app."""
    
    client_code = 100005
    
    response = requests.post(
        f"{heroku_url}/predict/",
        json={"client_code_1": client_code},
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    json_data = response.json()
    
    assert json_data['predictions']['prediction_1'] in ['Granted', 'Not Granted']
    assert 'probability_1' in json_data['predictions']
    assert 'values' in json_data['predictions']

    shap_values = json_data['predictions']['values']
    
    assert len(shap_values) > 0  

    assert isinstance(json_data['predictions']['probability_1'], float)
    
    assert isinstance(shap_values, list)
    
    dat_dict = json_data['dat']
    assert isinstance(dat_dict, dict)
   
    

    