




import pickle  # Or joblib if you're using that
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the trained model (replace 'model.pkl' with your actual model file)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Complete mappings for categorical features
team_mapping = {
    'Chennai Super Kings': 0,
    'Royal Challengers Bengaluru': 1,
    'Punjab Kings': 2,
    'Mumbai Indians': 3,
    'Kolkata Knight Riders': 4,
    'Rajasthan Royals': 5,
    'Sunrisers Hyderabad': 6,
    'Delhi Capitals': 7,
    'Lucknow Super Giants': 8,
    'Gujarat Titans': 9
}

city_mapping = {
    'Bangalore': 0,
    'Chennai': 1,
    'Delhi': 2,
    'Mumbai': 3,
    'Kolkata': 4,
    'Jaipur': 5,
    'Hyderabad': 6,
    'Chandigarh': 7,
    'Cape Town': 8,
    'Port Elizabeth': 9,
    'Durban': 10,
    'Centurion': 11,
    'East London': 12,
    'Johannesburg': 13,
    'Kimberley': 14,
    'Bloemfontein': 15,
    'Ahmedabad': 16,
    'Cuttack': 17,
    'Nagpur': 18,
    'Dharamsala': 19,
    'Visakhapatnam': 20,
    'Pune': 21,
    'Raipur': 22,
    'Ranchi': 23,
    'Abu Dhabi': 24,
    'Bengaluru': 25,
    'Indore': 26,
    'Dubai': 27,
    'Sharjah': 28,
    'Navi Mumbai': 29,
    'Lucknow': 30,
    'Guwahati': 31,
    'Mohali': 32
}

def map_feature(value, mapping, feature_name):
    try:
        return mapping[value]
    except KeyError:
        raise ValueError(f"Invalid value for {feature_name}: {value}")

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Expecting JSON input from the Flutter app
    try:
        # Extract features from data and convert them to numeric values
        features = [
            map_feature(data['batting_team'], team_mapping, 'batting_team'),
            map_feature(data['bowling_team'], team_mapping, 'bowling_team'),
            map_feature(data['city'], city_mapping, 'city'),
            int(data['runs_left']),
            int(data['balls_left']),
            int(data['wickets_left']),
            int(data['target_runs']),
            float(data['crr']),
            float(data['rrr'])
        ]
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Predict the probability of winning
    prediction_proba = model.predict_proba([features])[0][1]
    winning_percentage = prediction_proba * 100
    
    return jsonify({'result': winning_percentage})

    @app.route('/test', methods=['GET'])
   def test():
       return jsonify({"message": "API is working"}), 200
# if __name__ == '__main__':
    app.run()
