# Affinda Matching Code

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np

# Initialize vectorizer with parameters optimized for programming language names
vectorizer = TfidfVectorizer(
    lowercase=True,
    analyzer='word',
    ngram_range=(1, 2),
    max_features=1000,
    stop_words='english',
    token_pattern=r'(?u)\b\w+[\w\.+-]*\w*\b'  # Modified to handle special characters in language names
)

# Initialize classifier with optimized parameters
classifier = SGDClassifier(
    loss="modified_huber",
    max_iter=1000,
    tol=1e-4,
    random_state=42,
    eta0=0.1,
    learning_rate='constant',
    alpha=0.0001,
    class_weight='balanced',
    warm_start=True
)

# Create pipeline
model = Pipeline([
    ('vectorizer', vectorizer),
    ('classifier', classifier)
])

def load_initial_data(file_path='new.csv'):
    """
    Load initial data from CSV and train the model.
    """
    if os.path.exists(file_path):
        # Read the CSV file
        initial_data = pd.read_csv(file_path)
        print('start')
        print(initial_data['skill_variation'])
        print('end')
        # Clean the data
        initial_data['skill_variation'] = initial_data['skill_variation'].str.lower().str.strip()
        initial_data['base_skill'] = initial_data['base_skill'].str.lower().str.strip()
        
        # Remove duplicates
        initial_data = initial_data.drop_duplicates()
        
        # Create sample weights (give more weight to exact matches)
        sample_weights = np.ones(len(initial_data))
        exact_matches = initial_data['skill_variation'] == initial_data['base_skill']
        sample_weights[exact_matches] = 2.0
        print(exact_matches)



        # Train the model
        model.fit(initial_data['skill_variation'], initial_data['base_skill'], 
                 classifier__sample_weight=sample_weights)
        
        print("Initial model training complete.")
        print("\nAvailable base skills:", sorted(initial_data['base_skill'].unique()))
    else:
        print(f"{file_path} not found. Please provide the initial training data file.")

def predict_base_skills(skill_variations):
    """
    Predict base skills for input variations.
    Only returns predictions with a confidence score of 50% or higher.
    """
    if isinstance(skill_variations, str):
        skill_variations = [skill_variations]
    
    # Clean input
    skill_variations = [str(skill).lower().strip() for skill in skill_variations]
    
    # Get predictions and probabilities
    predictions = model.predict(skill_variations)
    probabilities = model.predict_proba(skill_variations)
    
    # Print predictions with confidence scores, filter by confidence
    results = []
    for skill, pred, probs in zip(skill_variations, predictions, probabilities):
        confidence = max(probs) * 100
        if confidence >= 50:  # Only include predictions with 50% confidence or higher
            print(f"Input Skill: '{skill}' -> Predicted Base Skill: '{pred}' (Confidence: {confidence:.2f}%)")
            results.append((skill, pred, confidence))
    
    return results


def save_model(file_path='skill_model.pkl'):
    """
    Save the trained model to a file.
    """
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}.")

def load_model(file_path='skill_model.pkl', initial_data_path='new.csv'):
    """
    Load a trained model from a file. If not found, retrain and save the model.
    """
    global model
    if os.path.exists(file_path):
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}.")
    else:
        print("Model file not found. Retraining the model...")
        load_initial_data(initial_data_path)  # Retrain using initial data
        save_model(file_path)  # Save the retrained model


if __name__ == "__main__":
    # Load and train the model with the CSV file
    load_initial_data('new.csv')
    

    
    # Save the model
    save_model()
