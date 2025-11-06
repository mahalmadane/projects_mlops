import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(url, json=data)
print(response.json())

