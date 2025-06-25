from utils.preprocess import load_and_clean_data
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split

# 读取并清洗数据
df = load_and_clean_data('../data/new.csv')

# 特征和标签
features = [
    'Lng', 'Lat', 'Cid', 'DOM', 'followers', 'square', 'livingRoom', 'drawingRoom',
    'kitchen', 'bathRoom', 'buildingType', 'constructionTime', 'renovationCondition',
    'buildingStructure', 'ladderRatio', 'elevator', 'fiveYearsProperty', 'subway',
    'district', 'communityAverage', 'year', 'month', 'quarter'
]
label = 'price'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[label], test_size=0.2, random_state=42)

# 转为 LGB 数据格式
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# 训练参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'verbosity': -1
}

# 正确的 early stopping 和日志
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[valid_data],
    callbacks=[
        lgb.early_stopping(10),
        lgb.log_evaluation(10)
    ]
)

# 保存模型
joblib.dump(model, '../model/price_model.pkl')
print("✅ 模型训练完成并保存！")
