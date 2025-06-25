import pandas as pd


def load_and_clean_data(filepath):
    df = pd.read_csv(filepath, encoding='gb2312', low_memory=False)
    df['tradeTime'] = pd.to_datetime(df['tradeTime'], errors='coerce')
    df = df.drop(columns=['url'], errors='ignore')

    df.fillna({
        'elevator': 0,
        'subway': 0,
        'fiveYearsProperty': 1,
        'ladderRatio': df['ladderRatio'].mean()
    }, inplace=True)

    #  强制转换为数值类型
    cols_to_convert = ['livingRoom', 'drawingRoom', 'bathRoom', 'constructionTime']
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 时间特征
    df['year'] = df['tradeTime'].dt.year
    df['month'] = df['tradeTime'].dt.month
    df['quarter'] = df['tradeTime'].dt.quarter

    # 清除因非法类型转 float 造成的 NaN
    df.dropna(subset=cols_to_convert, inplace=True)

    return df