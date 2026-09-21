from flask import Flask, render_template, jsonify, request
from pathlib import Path

from model import RainfallRiskModel
from fusion import load_all_data, build_dashboard

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

LOCATIONS = {

    "Guntur": {
        "latitude": 16.3067,
        "longitude": 80.4365
    },

    "Vijayawada": {
        "latitude": 16.5062,
        "longitude": 80.6480
    },

    "Amaravati": {
        "latitude": 16.5730,
        "longitude": 80.3575
    },

    "Visakhapatnam": {
        "latitude": 17.6868,
        "longitude": 83.2185
    },

    "Hyderabad": {
        "latitude": 17.3850,
        "longitude": 78.4867
    },

    "Chennai": {
        "latitude": 13.0827,
        "longitude": 80.2707
    },

    "Bengaluru": {
        "latitude": 12.9716,
        "longitude": 77.5946
    },

    "Mumbai": {
        "latitude": 19.0760,
        "longitude": 72.8777
    },

    "Delhi": {
        "latitude": 28.6139,
        "longitude": 77.2090
    },

    "Kolkata": {
        "latitude": 22.5726,
        "longitude": 88.3639
    },

    "Pune": {
        "latitude": 18.5204,
        "longitude": 73.8567
    },

    "Ahmedabad": {
        "latitude": 23.0225,
        "longitude": 72.5714
    },

    "Jaipur": {
        "latitude": 26.9124,
        "longitude": 75.7873
    },

    "Lucknow": {
        "latitude": 26.8467,
        "longitude": 80.9462
    },

    "Bhopal": {
        "latitude": 23.2599,
        "longitude": 77.4126
    },

    "Indore": {
        "latitude": 22.7196,
        "longitude": 75.8577
    },

    "Nagpur": {
        "latitude": 21.1458,
        "longitude": 79.0882
    },

    "Patna": {
        "latitude": 25.5941,
        "longitude": 85.1376
    },

    "Ranchi": {
        "latitude": 23.3441,
        "longitude": 85.3096
    },

    "Bhubaneswar": {
        "latitude": 20.2961,
        "longitude": 85.8245
    },

    "Raipur": {
        "latitude": 21.2514,
        "longitude": 81.6296
    },

    "Chandigarh": {
        "latitude": 30.7333,
        "longitude": 76.7794
    },

    "Dehradun": {
        "latitude": 30.3165,
        "longitude": 78.0322
    },

    "Srinagar": {
        "latitude": 34.0837,
        "longitude": 74.7973
    },

    "Jammu": {
        "latitude": 32.7266,
        "longitude": 74.8570
    },

    "Kochi": {
        "latitude": 9.9312,
        "longitude": 76.2673
    },

    "Thiruvananthapuram": {
        "latitude": 8.5241,
        "longitude": 76.9366
    },

    "Coimbatore": {
        "latitude": 11.0168,
        "longitude": 76.9558
    },

    "Madurai": {
        "latitude": 9.9252,
        "longitude": 78.1198
    },

    "Mysuru": {
        "latitude": 12.2958,
        "longitude": 76.6394
    },

    "Surat": {
        "latitude": 21.1702,
        "longitude": 72.8311
    },

    "Jodhpur": {
        "latitude": 26.2389,
        "longitude": 73.0243
    },

    "Varanasi": {
        "latitude": 25.3176,
        "longitude": 82.9739
    },

    "Agra": {
        "latitude": 27.1767,
        "longitude": 78.0081
    }
}

# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# Load and train ML model
# --------------------------------------------------

print("====================================")
print("AI Rainfall Early Warning System")
print("====================================")

print("Loading training dataset...")

model = RainfallRiskModel(
    DATA_DIR / "training.csv"
)

print("Training Random Forest model...")

accuracy = model.train()

print(
    f"Model training completed."
)

print(
    f"Validation Accuracy: {accuracy * 100:.2f}%"
)


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Dashboard API
# --------------------------------------------------

@app.route(
    "/api/dashboard",
    methods=["GET"]
)
def dashboard():

    try:

        city = request.args.get("city", "Guntur").strip()

        if city not in LOCATIONS:
            return jsonify({
                "error": "Unknown city.",
                "available_cities": list(LOCATIONS)
            }), 400

        data = load_all_data(
            DATA_DIR
        )

        result = build_dashboard(
            data,
            model,
            city=city,
            location=LOCATIONS[city]
        )

        return jsonify(result)

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.route(
    "/api/health",
    methods=["GET"]
)
def health():

    return jsonify({
        "status": "running",
        "model_ready": model.model_ready,
        "training_rows": model.training_rows
    })


# --------------------------------------------------
# Run Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
