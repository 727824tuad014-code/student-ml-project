from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# load trained model
model = joblib.load("model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        hours = float(data['hours'])
        attendance = float(data['attendance'])
        sleep = float(data['sleep'])
    except:
        return jsonify({"error": "Invalid input"}), 400

    # constraints
    if not (0 <= hours <= 12):
        return jsonify({"error": "Study hours must be between 0 and 12"}), 400

    if not (0 <= attendance <= 100):
        return jsonify({"error": "Attendance must be between 0 and 100"}), 400

    if not (0 <= sleep <= 12):
        return jsonify({"error": "Sleep hours must be between 0 and 12"}), 400

    result = model.predict([[hours, attendance, sleep]])[0]

    if result >= 75:
        level = "Excellent"
    elif result >= 50:
        level = "Average"
    else:
        level = "Needs Improvement"

    return jsonify({
        "prediction": round(result, 2),
        "level": level
    })

if __name__ == '__main__':
    app.run(debug=True)