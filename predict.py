import joblib
import pandas as pd

# Load the model and state
def load_model(model_file='skill_model.pkl', state_file='model_state.pkl'):
    model = joblib.load(model_file)
    return model

def predict_base_skills(model, skill_variations):
    if isinstance(skill_variations, str):
        skill_variations = [skill_variations]

    # Clean input
    skill_variations = [str(skill).lower().strip() for skill in skill_variations]

    # Get predictions and probabilities
    predictions = model.predict(skill_variations)
    probabilities = model.predict_proba(skill_variations)

    # Print predictions with confidence scores
    results = []
    for skill, pred, probs in zip(skill_variations, predictions, probabilities):
        confidence = max(probs) * 100
        print(f"Input Skill: '{skill}' -> Predicted: '{pred}' (Confidence: {confidence:.2f}%)")
        results.append((skill, pred, confidence))
    
    return results

if __name__ == "__main__":
    # Load the trained model
    model = load_model()

    # Sample skills to predict
    test_skills = [
        'ts',
        'Node.JS (Javascript Library)',
        'next js'
    ]
    # Make predictions
    print("\nTesting model predictions on sample inputs:")
    predict_base_skills(model, test_skills)