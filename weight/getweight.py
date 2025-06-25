import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

# --------------------
# 1. 读取数据
# --------------------
df = pd.read_csv('../data/new.csv', encoding='gb2312', low_memory=False)

# --------------------
# 2. 清洗数据
# --------------------
# 删除无关列和共线性强的列
drop_cols = ['url', 'id', 'tradeTime', 'price', 'communityAverage']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 替换非法值 '#NAME?' 为 NaN
df.replace("#NAME?", pd.NA, inplace=True)

# 指定类别字段（用于独热编码）
categorical_features = ['floor', 'renovationCondition', 'buildingStructure']

# 将非类别字段统一转为数值
for col in df.columns:
    if col not in categorical_features + ['totalPrice']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 填补缺失值为 0
df = df.fillna(0)

# --------------------
# 3. 准备特征与目标
# --------------------
target = 'totalPrice'
X = df.drop(columns=[target])
y = df[target]

# 再次确认类别特征是否都在数据中
categorical_features = [col for col in categorical_features if col in X.columns]
numerical_features = [col for col in X.columns if col not in categorical_features]

# --------------------
# 4. 特征预处理 + 模型管道
# --------------------
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=0.1))
])

# --------------------
# 5. 拟合模型
# --------------------
pipeline.fit(X, y)

# --------------------
# 6. 输出权重结果
# --------------------
# 获取处理后的特征名
feature_names = numerical_features + list(
    pipeline.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out(categorical_features)
)

# 获取系数
coefficients = pipeline.named_steps['regressor'].coef_

# 整理输出结果
importance = pd.DataFrame({
    'Feature': feature_names,
    'Weight': coefficients
}).sort_values(by='Weight', ascending=False)

# 打印前10个重要特征
print("Top 10 most important features:")
print(importance.head(10))

# 保存所有结果到 CSV
importance.to_csv("feature_weights.csv", index=False)
print("权重结果已保存为 feature_weights.csv")
