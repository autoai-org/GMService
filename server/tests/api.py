import requests

def test_fetch_models():
    response = requests.get("http://localhost:8000/api/models")
    assert response.status_code == 200
    assert len(response.json()) > 0

def test_predict():
    query = {
        'model': 'mix_0.5 * ../.cache/models/pythia-dolly-2000/_0.5 * ../.cache/models/pythia-sharegpt-6000/_',
        'body': {
            "prompt": "<human>: What can you do?\n<bot>:",
            "max_tokens": 128,
            "temperature": 0.9,
            "top_p": 0.9,
            "n": 1,
        },
    }
    response = requests.post("http://localhost:8000/api/predict", json=query)
    assert response.status_code == 200

if __name__ == "__main__":
    test_fetch_models()
    test_predict()