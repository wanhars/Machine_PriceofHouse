import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "Lng": 116.35, "Lat": 39.93, "Cid": 1, "DOM": 30, "followers": 200,
    "square": 90, "livingRoom": 2, "drawingRoom": 1, "kitchen": 1, "bathRoom": 1,
    "buildingType": 4.0, "constructionTime": 2005, "renovationCondition": 2,
    "buildingStructure": 1, "ladderRatio": 0.5, "elevator": 1,
    "fiveYearsProperty": 1, "subway": 1, "district": 10, "communityAverage": 72000,
    "year": 2024, "month": 1, "quarter": 2
}

response = requests.post(url, json=payload)
print(response.json())