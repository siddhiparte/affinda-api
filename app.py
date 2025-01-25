from flask import Flask, request, jsonify
from flask_cors import CORS
from relevance import load_model, predict_base_skills
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)
CORS(app)

# Load the model once when the application starts
try:
    load_model()  # Ensure this is called to load your trained model
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict_skills():
    """Endpoint to predict base skills from input skill variations"""
    try:
        data = request.get_json()

        # Check if the 'skills' key is present and valid
        if not data or 'skills' not in data:
            return jsonify({"error": "Invalid request format. Please provide 'skills' array in JSON"}), 400
        
        skills = data['skills']

        if len(skills) == 0:
            # Return empty array if skills is empty
            return jsonify({"predictions": []}), 200
        
        # Debug: Log the received skills
        logger.info(f"Received skills: {skills}")

        # Make predictions using the imported function
        results = predict_base_skills(skills)

        # Format response
        response = {
            "predictions": [
                {
                    "base_skill": base_skill,
                }
                for input_skill, base_skill, confidence in results
            ]
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
