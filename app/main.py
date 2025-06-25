from flask import Flask, request, jsonify
from predictor import predict_price

app = Flask(__name__)
@app.route('/')
def home():
    return '🏠 欢迎使用房价预测 API，请使用 POST /predict 提交数据'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        price = predict_price(data)
        recommendation = '✅ 推荐购入' if price < data['communityAverage'] * 0.9 else (
            '❌ 谨慎购入' if price > data['communityAverage'] * 1.1 else '⚖️ 观望')
        return jsonify({
            '预测价格': round(price, 2),
            '推荐': recommendation
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
