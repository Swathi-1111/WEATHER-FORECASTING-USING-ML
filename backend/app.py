from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import requests
import os
import logging
import traceback
from dotenv import load_dotenv
from gtts import gTTS

# Load environment variables from .env file
load_dotenv()

# === Setup Logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# === Config ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
STATIC_DIR = os.path.join(BASE_DIR, "../static")
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")

# Get OpenWeather API key from environment variable
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    raise ValueError("API key not found. Please set OPENWEATHER_API_KEY in your .env file")

# Set to False to use real OpenWeatherMap API
use_fake_data = False  # Using real API data

# === Ensure directories exist ===
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# === Load model & scaler ===
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info("ML model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or scaler: {str(e)}")
    raise

# === Setup Flask ===
app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)
CORS(app)

# Audio features disabled
logger.info("Audio features are disabled in this version")

@app.route("/")
def home():
    """Serve the main application page"""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests and integrate all services"""
    try:
        # Get city input
        city = request.form.get("city")
        if not city:
            logger.warning("Request missing city parameter")
            return jsonify({"error": "City name is required."}), 400

        # Start prediction process
        if use_fake_data:
            logger.info(f"[SIMULATION] Using fake data instead of API call for {city}")
            weather_condition = "rain"
            temperature = 22.3
            humidity = 92
            wind_speed = 3.8
            pressure = 1001
            icon = "09d"
            precipitation = 3.2
        else:
            # Get real weather data
            logger.info(f"Fetching weather data for '{city}'")
            weather_data = fetch_weather_data(city)
            
            if "error" in weather_data:
                return jsonify(weather_data), weather_data.get("status_code", 500)
            
            # Extract weather parameters
            temperature = weather_data["main"]["temp"]
            humidity = weather_data["main"]["humidity"]
            wind_speed = weather_data["wind"]["speed"]
            pressure = weather_data["main"]["pressure"]
            weather_condition = weather_data["weather"][0]["main"].lower()
            icon = weather_data["weather"][0].get("icon", "01d")
            
            # Process precipitation data
            precipitation = calculate_precipitation(weather_data)

        logger.info(f"Weather data - Temp: {temperature}°C, Humidity: {humidity}%, Wind: {wind_speed}m/s")

        # Make ML prediction
        try:
            features = np.array([[temperature, humidity, wind_speed, pressure]])
            scaled = scaler.transform(features)
            prediction = model.predict(scaled)
            
            # Process prediction results
            predicted_values = process_prediction(prediction)
            predicted_temp = predicted_values["temperature"]
            predicted_precip = predicted_values["precipitation"]
            
            logger.info(f"Prediction - Temp: {predicted_temp}°C, Precipitation: {predicted_precip}mm")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return jsonify({"error": "Failed to process prediction"}), 500

        # Choose appropriate video based on ML PREDICTION
        video_file = choose_weather_video(weather_condition, temperature, predicted_temp, predicted_precip)

        # Generate TTS audio
        try:
            tts_text = f"The weather in {city} is {weather_condition} with a temperature of {round(temperature, 1)} degrees Celsius. Humidity is {humidity} percent. Tomorrow's forecast is {weather_condition} with a temperature of {round(predicted_temp, 1)} degrees Celsius."
            audio_filename = f"{city.lower().replace(' ', '_')}.mp3"
            audio_path = os.path.join(AUDIO_DIR, audio_filename)
            
            # Generate only if doesn't exist or we want fresh audio
            tts = gTTS(text=tts_text, lang='en')
            tts.save(audio_path)
            logger.info(f"Generated TTS audio for {city} at {audio_path}")
        except Exception as e:
            logger.error(f"Failed to generate TTS audio: {str(e)}")

        # Create response object matching frontend expectations
        response_data = {
            "city": city.title(),
            "weather_condition": weather_condition.capitalize(),
            "temperature": round(temperature, 1),
            "humidity": humidity,
            "wind_speed": round(wind_speed, 1),
            "pressure": pressure,
            "icon": icon,
            "video": video_file,
            "coord": weather_data.get("coord", {"lat": 0, "lon": 0}),
            "predicted_temperature": round(predicted_temp, 1),
            "predicted_precipitation": round(predicted_precip, 1),
            "predicted_condition": weather_condition.capitalize(),
            "audio": f"/static/audio/{audio_filename}"
        }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Unhandled exception in prediction endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500

def fetch_weather_data(city):
    """Fetch weather data from OpenWeather API with robust error handling"""
    if use_fake_data:
        logger.info(f"Using fake weather data for {city}")
        # Return comprehensive sample data for testing
        return {
            "coord": {"lon": -0.13, "lat": 51.51},
            "weather": [
                {
                    "id": 800,
                    "main": "Clear",
                    "description": "clear sky",
                    "icon": "01d"
                }
            ],
            "base": "stations",
            "main": {
                "temp": 293.25,
                "feels_like": 292.5,
                "temp_min": 290.15,
                "temp_max": 295.15,
                "pressure": 1012,
                "humidity": 60,
                "sea_level": 1019,
                "grnd_level": 1011
            },
            "visibility": 10000,
            "wind": {
                "speed": 3.1,
                "deg": 240,
                "gust": 5.1
            },
            "clouds": {
                "all": 0
            },
            "dt": 1600000000,
            "sys": {
                "type": 1,
                "id": 1414,
                "country": "GB",
                "sunrise": 1599950000,
                "sunset": 1599990000
            },
            "timezone": 0,
            "id": 2643743,
            "name": city,
            "cod": 200
        }
    
    # If not using fake data, try to fetch from OpenWeather API
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    for attempt in range(3):
        try:
            logger.info(f"Weather API request attempt {attempt+1}")
            response = requests.get(url, timeout=15)
            
            # Handle common errors
            if response.status_code == 401:
                logger.error("OpenWeather API key unauthorized")
                return {"error": "Weather API authorization failed", "status_code": 401}
                
            if response.status_code == 404:
                logger.warning(f"City '{city}' not found")
                return {"error": f"City '{city}' not found", "status_code": 404}
                
            response.raise_for_status()
            data = response.json()
            logger.info("OpenWeather API request successful")
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed (attempt {attempt+1}): {str(e)}")
            if attempt == 2:  # If this was the last attempt
                return {"error": f"Failed to fetch weather data: {str(e)}", "status_code": 500}
            time.sleep(1)  # Wait before retrying
                
        except requests.exceptions.Timeout:
            logger.warning(f"OpenWeather API timeout on attempt {attempt+1}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # Exponential backoff
            logger.error(f"OpenWeather API request failed: {str(e)}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
    
    # If we get here, all attempts failed
    return {"error": "Failed to retrieve weather data after multiple attempts", "status_code": 503}

def calculate_precipitation(weather_data):
    """Extract and calculate precipitation from weather data"""
    try:
        rain_data = weather_data.get("rain", {})
        snow_data = weather_data.get("snow", {})
        
        # Handle different data types
        if isinstance(rain_data, dict):
            rain = float(rain_data.get("1h", 0))
        else:
            rain = 0
            
        if isinstance(snow_data, dict):
            snow = float(snow_data.get("1h", 0))
        else:
            snow = 0
            
        precipitation = round(rain + snow, 2)
        return precipitation
    except Exception as e:
        logger.warning(f"Error calculating precipitation: {str(e)}")
        return 0.0

def process_prediction(prediction):
    """Process model prediction into standard format"""
    try:
        if isinstance(prediction[0], (list, np.ndarray)) and len(prediction[0]) >= 2:
            predicted_temp = round(float(prediction[0][0]), 2)
            predicted_precip = round(float(prediction[0][1]), 2)
        elif isinstance(prediction[0], (int, float, np.number)):
            predicted_temp = round(float(prediction[0]), 2)
            predicted_precip = 0.0
        else:
            logger.warning(f"Unexpected prediction format: {type(prediction[0])}")
            predicted_temp = 0.0
            predicted_precip = 0.0
            
        return {
            "temperature": predicted_temp,
            "precipitation": predicted_precip
        }
    except Exception as e:
        logger.error(f"Error processing prediction: {str(e)}")
        return {
            "temperature": 0.0,
            "precipitation": 0.0
        }

def choose_weather_video(condition, temp, pred_temp, pred_precip):
    """Select appropriate video based on ML-Predicted weather conditions"""
    condition = condition.lower()
    
    # ML PRIORITY 1: High Precipitation Forecast
    if pred_precip > 0.4:
        return 'rain.mp4'
    
    # ML PRIORITY 2: Thermal Forecast
    if pred_temp > 30:
        return 'sunny.mp4'  # Predicted Heatwave
    elif pred_temp < 15:
        return 'cloudy.mp4' # Predicted Cold/Cloudy
        
    # FALLBACK: Current Conditions
    video_map = {
        "clear": "sunny.mp4",
        "clouds": "cloudy.mp4",
        "rain": "rain.mp4",
        "drizzle": "rain.mp4",
        "snow": "snow.mp4",
        "thunderstorm": "rain.mp4",
        "mist": "cloudy.mp4",
        "fog": "cloudy.mp4",
        "haze": "cloudy.mp4"
    }
    return video_map.get(condition, "default.mp4")

def serve_static(filename):
    """Serve static files"""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/test')
def test_apis():
    """Test endpoint to verify all API connections"""
    try:
        # Test OpenWeather API
        test_city = "London"
        weather = fetch_weather_data(test_city)
        if "error" in weather:
            return jsonify({
                "openweather_status": "error",
                "openweather_error": weather["error"]
            }), 500
            
        # Test ML model
        try:
            test_features = np.array([[20, 70, 5, 1013]])
            scaled = scaler.transform(test_features)
            prediction = model.predict(scaled)
            return jsonify({
                "openweather_status": "ok",
                "ml_model_status": "ok"
            })
        except Exception as e:
            return jsonify({
                "openweather_status": "ok",
                "ml_model_status": f"error: {str(e)}"
            }), 500
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "version": "1.0.0"})

if __name__ == "__main__":
    logger.info("Weather Prediction App starting up...")
    app.run(debug=True)