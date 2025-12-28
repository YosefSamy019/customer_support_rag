import requests


def endpoint_call(url: str, headers: dict, payload: dict) -> dict:
    res = requests.post(url, headers=headers, json=payload)
    answer = res.json()
    return answer
