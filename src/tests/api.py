import requests

def test_fetch_models():
    response = requests.get("http://localhost:8000/api/models")
    print(response.json())
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_predict():
    query = {
        'model': 'pythia-openalign',
        'body': {
            "prompt": "<human>: hello\n<bot>:",
            "max_tokens": 10,
            "temperature": 0.9,
            "top_p": 0.9,
            "n": 1,
        },
    }
    response = requests.post("http://localhost:8000/api/predict", json=query)
    print(response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    test_fetch_models()
    test_predict()