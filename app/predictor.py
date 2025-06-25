import joblib
import numpy as np

model = joblib.load('../model/price_model.pkl')

def predict_price(input_data: dict):
    features_order = [
        'Lng', 'Lat', 'Cid', 'DOM', 'followers', 'square', 'livingRoom', 'drawingRoom',
        'kitchen', 'bathRoom', 'buildingType', 'constructionTime', 'renovationCondition',
        'buildingStructure', 'ladderRatio', 'elevator', 'fiveYearsProperty', 'subway',
        'district', 'communityAverage', 'year', 'month', 'quarter'
    ]
    X = np.array([[input_data[feature] for feature in features_order]])
    return float(model.predict(X)[0])
