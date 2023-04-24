import requests


def test_fetch_models():
    response = requests.get("http://localhost:8000/api/models")
    print(response.json())
    assert response.status_code == 200
    assert len(response.json()) > 0


def test_mix():
    query = {
        "models": ["../.cache/models/pythia-dolly-2000/", "../.cache/models/pythia-sharegpt-6000/"],
        "weights": [0.5, 0.5]
    }
    response = requests.post("http://localhost:8000/action/mix", json=query)
    print(response.json())
    assert response.status_code == 200


if __name__ == "__main__":
    test_fetch_models()
    test_mix()
