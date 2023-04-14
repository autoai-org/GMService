import requests


def test_random_chat_single():
    query = {
        "model": "random",
        "dialogs": [
            {"role": "USER", "text": "hello"},
        ],
        'body': {
            "max_tokens": 10,
            "temperature": 0.9,
            "top_p": 0.9,
            "n": 1,
        },
    }
    response = requests.post("http://localhost:8000/api/chat", json=query)
    print(response.json())
    assert response.status_code == 200

if __name__ == "__main__":
    test_random_chat_single()