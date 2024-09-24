import pickle  # Or joblib if you're using that
from flask import Flask, request, jsonify

# Load the trained model (replace 'model.pkl' with your actual model file)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON input from the Flutter app
    # Extract features from data and predict using the model
    features = [data['batting_team'], data['bowling_team'], data['city'], data['runs_left'],
                data['balls_left'], data['wickets_left'], data['target_runs'], data['crr'], data['rrr']]
    
    prediction = model.predict([features])
    
    return jsonify({'result': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
