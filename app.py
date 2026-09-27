from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import json
import os
import joblib
import pandas as pd
import requests
import gdown
from PIL import Image
import io
import random

app = Flask(__name__)

CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
if CORS_ORIGINS == '*':
    cors_origins = ['*']
else:
    cors_origins = [origin.strip() for origin in CORS_ORIGINS.split(',') if origin.strip()]

CORS(app, resources={r"/*": {"origins": cors_origins}}, supports_credentials=True)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_DIR = BASE_DIR / 'model'
MODEL_DIR.mkdir(exist_ok=True)

SOIL_DATA_PATH = MODEL_DIR / 'state_soil_data.csv'
MODEL_PATH = MODEL_DIR / 'crop_recommendation_model_new.joblib'
CROP_RECOMMENDATION_DATA_PATH = MODEL_DIR / 'Crop_recommendation_with_soil.csv'
MARKET_PRICES_PATH = BASE_DIR / 'market_prices.json'
MODEL_URL = os.getenv('MODEL_URL', '')


def load_model_from_disk():
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            print(f"Existing model file is invalid or corrupted: {exc}")
            try:
                MODEL_PATH.unlink()
            except OSError:
                pass

    if MODEL_URL:
        try:
            gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False)
            if MODEL_PATH.exists():
                try:
                    return joblib.load(MODEL_PATH)
                except Exception as exc:
                    print(f"Downloaded model is invalid or corrupted: {exc}")
        except Exception as exc:
            print(f"Model download failed: {exc}")
    return None


model = load_model_from_disk()

try:
    if SOIL_DATA_PATH.exists():
        state_soil_data = pd.read_csv(SOIL_DATA_PATH)
    else:
        training_data_path = PROJECT_ROOT / 'model' / 'clean_crop_dataset_expanded.csv'
        if not training_data_path.exists():
            training_data_path = BASE_DIR / 'model' / 'clean_crop_dataset_expanded.csv'
        if training_data_path.exists():
            state_soil_data = pd.read_csv(training_data_path)
            state_soil_data = state_soil_data.groupby('state', as_index=False).agg(
                N=('N', 'mean'),
                P=('P', 'mean'),
                K=('K', 'mean')
            )
        else:
            state_soil_data = pd.DataFrame(columns=['state', 'N', 'P', 'K'])
except Exception as exc:
    print(f"Could not load soil metadata: {exc}")
    state_soil_data = pd.DataFrame(columns=['state', 'N', 'P', 'K'])

if 'state' in state_soil_data.columns:
    all_states = state_soil_data['state'].dropna().astype(str).unique().tolist()
else:
    all_states = []

try:
    if CROP_RECOMMENDATION_DATA_PATH.exists():
        crop_data_for_soil_types = pd.read_csv(CROP_RECOMMENDATION_DATA_PATH)
        if 'soil_type' in crop_data_for_soil_types.columns:
            all_soil_types = sorted(crop_data_for_soil_types['soil_type'].dropna().unique().tolist())
        else:
            all_soil_types = ['Clay', 'Alluvial', 'Laterite', 'Loamy', 'Sandy']
    else:
        all_soil_types = ['Clay', 'Alluvial', 'Laterite', 'Loamy', 'Sandy']
except Exception as exc:
    print(f"Could not load soil type metadata: {exc}")
    all_soil_types = ['Clay', 'Alluvial', 'Laterite', 'Loamy', 'Sandy']


@app.route('/')
def home():
    return "Welcome to the Crop Recommendation API!"

@app.route('/get_soil_types', methods=['GET']) # New endpoint
def get_soil_types():
    return jsonify(list(all_soil_types))

@app.route('/get_states', methods=['GET']) # New endpoint
def get_states():
    return jsonify(all_states)

@app.route('/recommend_crop', methods=['POST'])
def recommend_crop():
    """
    Takes state, city and ph data and returns a crop suggestion.
    """
    try:
        data = request.get_json()
        print(f"Received data for recommend_crop: {data}") # Log incoming data
        state = data.get('state')
        city = data.get('city')
        ph = data.get('ph')
        soil_type_from_image = data.get('soil_type_from_image') # New: Get soil type from image if available
        soil_type_from_dropdown = data.get('soil_type_from_dropdown') # New: Get soil type from dropdown if available

        if model is None:
            return jsonify({"error": "Model is not available. Please upload the model file or set MODEL_URL in the backend environment."}), 503

        if not all([state, city, ph]):
            print("Missing state, city or ph in request.") # Log missing data
            return jsonify({"error": "Missing state, city or ph"}), 400

        # Removed weather data fetching
        temperature = 0 # Placeholder
        humidity = 0 # Placeholder
        rainfall = 0 # Placeholder

        # Determine soil type: dropdown > image > Bhuvan API > fallback
        n, p, k = None, None, None
        soil_type = None

        if soil_type_from_dropdown:
            soil_type = soil_type_from_dropdown
            print(f"Soil type from dropdown: {soil_type}")
        elif soil_type_from_image:
            soil_type = soil_type_from_image
            print(f"Soil type from image: {soil_type}")
        else:
            try:
                bhuvan_url = f"http://bhuvan.nrsc.gov.in/search/v1/json/{city}/0-2"
                print(f"Attempting to fetch soil data from Bhuvan API: {bhuvan_url}")
                bhuvan_response = requests.get(bhuvan_url)
                bhuvan_data = bhuvan_response.json()
                # --- IMPORTANT ---
                # The following parsing is a guess based on a possible response structure.
                # You might need to inspect the actual response and adjust the parsing logic.
                n = bhuvan_data['features'][0]['properties']['N']
                p = bhuvan_data['features'][0]['properties']['P']
                k = bhuvan_data['features'][0]['properties']['K']
                soil_type = bhuvan_data['features'][0]['properties'].get('soil_type', 'Unknown') # Assuming soil_type is available
                print(f"Soil data from Bhuvan API: N={n}, P={p}, K={k}, Soil Type={soil_type}")
            except Exception as e:
                print(f"Error fetching soil data from Bhuvan API: {e}. Falling back to state average.")
                # Fallback to state average
                state_data = state_soil_data[state_soil_data['state'].str.lower() == state.lower()]
                if state_data.empty:
                    print(f"Could not find soil data for state: {state}")
                    return jsonify({"error": f"Could not find soil data for state: {state}"}), 404
                n = state_data['N'].iloc[0]
                p = state_data['P'].iloc[0]
                k = state_data['K'].iloc[0]
                soil_type = 'Unknown' # Default soil type if Bhuvan fails and no specific state soil type is available
                print(f"Soil data from state average: N={n}, P={p}, K={k}, Soil Type={soil_type}")

        # Create a dataframe for prediction
        input_data = {
            'N': [n], 
            'P': [p], 
            'K': [k], 
            'temperature': [temperature], # Keep for model input, but set to 0
            'humidity': [humidity],     # Keep for model input, but set to 0
            'ph': [ph], 
            'rainfall': [rainfall],     # Keep for model input, but set to 0
            'soil_type': [soil_type]
        }
        df = pd.DataFrame(input_data)
        print(f"Input data for prediction: {input_data}")

        # Perform one-hot encoding for 'soil_type'
        # Create dummy columns for all possible soil types encountered during training
        for st in all_soil_types:
            if st != 'Unknown': # 'Unknown' is not a dummy variable
                df[f'soil_type_{st}'] = (df['soil_type'] == st).astype(int)
        
        # Drop the original 'soil_type' column
        df = df.drop('soil_type', axis=1)

        # Ensure all columns from training are present, fill missing with 0
        # This is crucial because not all soil types might be present in a single prediction request
        # And the model expects all one-hot encoded columns it was trained on.
        # Get the columns the model was trained on
        model_features = model.feature_names_in_ 
        
        # Reindex the DataFrame to match the training columns, filling missing with 0
        df = df.reindex(columns=model_features, fill_value=0)
        print(f"DataFrame before prediction: {df}")

        # Make prediction
        prediction = model.predict(df)
        print(f"Prediction: {prediction[0]}")
        
        response = jsonify({
            "crop_suggestion": prediction[0],
            "confidence_score": 0.95  # Placeholder, will be updated later
        })
        print(f"Sending response: {response.json}") # Log outgoing response
        return response
    except Exception as e:
        print(f"An unexpected error occurred in recommend_crop: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

@app.route('/classify_soil_image', methods=['POST'])
def classify_soil_image():
    """
    Accepts an uploaded image and returns a dummy soil type.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']
    # You can add image processing here if needed, e.g., to resize or validate
    # For now, we'll just simulate a classification
    try:
        img = Image.open(io.BytesIO(file.read()))
        # Dummy classification: return a random soil type from the list
        predicted_soil_type = random.choice([st for st in all_soil_types if st != 'Unknown'])
        return jsonify({
            "soil_type": predicted_soil_type,
            "confidence_score": 0.80 # Dummy confidence
        })
    except Exception as e:
        return jsonify({"error": f"Error processing image: {e}"}), 500


@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Returns a yield and profit estimate.
    """
    data = request.get_json()
    crop = data.get('crop')
    with open(MARKET_PRICES_PATH, 'r') as f:
        market_prices = json.load(f)

    price = market_prices.get(crop.lower(), 0)
    # Dummy logic
    yield_estimate = 5 # tons/hectare
    profit_estimate = yield_estimate * price * 10 # dummy calculation
    return jsonify({
        "yield_estimate": f"{yield_estimate} tons/hectare",
        "profit_estimate": f"${profit_estimate}/hectare"
    })

@app.route('/disease_detect', methods=['POST'])
def disease_detect():
    """
    Accepts an uploaded image and returns disease information.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    # Dummy response
    return jsonify({
        "disease_name": "Leaf Blight",
        "confidence_score": 0.92,
        "remedy": "Apply fungicide"
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
