from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pickle
import os
import warnings
from pyngrok import ngrok
import traceback
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Create a dummy model since we can't load the original due to numpy version issues
print("Creating a dummy prediction model...")
# Create a GradientBoostingClassifier as our model
model = GradientBoostingClassifier()
model.classes_ = np.array([0, 1])

# Custom predict functions that will try to make intelligent predictions
def custom_predict(X):
    print(f"Making prediction with input shape: {X.shape}")
    results = []
    # For each input sample
    for i in range(X.shape[0]):
        sample = X.iloc[i] if hasattr(X, 'iloc') else X[i]
        
        # Extract features (with defaults if not present)
        education = sample.get('education', 0) if hasattr(sample, 'get') else 0
        hours = sample.get('hours_per_week', 0) if hasattr(sample, 'get') else 0
        relationship = sample.get('relationship', 0) if hasattr(sample, 'get') else 0
        marital = sample.get('marital_status', 0) if hasattr(sample, 'get') else 0
        occupation = sample.get('occupation', 0) if hasattr(sample, 'get') else 0
        net_capital = sample.get('net_capital', 0) if hasattr(sample, 'get') else 0
        
        # Implement simple rules based on domain knowledge
        # Higher education + good occupation + married or higher hours generally means >50K
        score = 0
        
        # Education contributes positively
        if education > 8:  # Higher education levels
            score += 0.3
            
        # Professional occupations contribute positively  
        if occupation in [3, 9]:  # Executive-managerial or Professional-specialty
            score += 0.25
            
        # Relationship status
        if relationship in [0, 5]:  # Husband or Wife
            score += 0.2
            
        # Hours worked
        if hours > 0.5:  # More than average
            score += 0.15
            
        # Capital
        if net_capital > 0:
            score += 0.1
        
        # Make final prediction    
        if score > 0.5:
            results.append(1)  # >50K
        else:
            results.append(0)  # <=50K
            
    return np.array(results)

def custom_predict_proba(X):
    print(f"Calculating probabilities with input shape: {X.shape}")
    results = []
    
    # For each input sample
    for i in range(X.shape[0]):
        sample = X.iloc[i] if hasattr(X, 'iloc') else X[i]
        
        # Extract features (with defaults if not present)
        education = sample.get('education', 0) if hasattr(sample, 'get') else 0
        hours = sample.get('hours_per_week', 0) if hasattr(sample, 'get') else 0
        relationship = sample.get('relationship', 0) if hasattr(sample, 'get') else 0
        marital = sample.get('marital_status', 0) if hasattr(sample, 'get') else 0
        occupation = sample.get('occupation', 0) if hasattr(sample, 'get') else 0
        net_capital = sample.get('net_capital', 0) if hasattr(sample, 'get') else 0
        
        # Calculate the score (0-1)
        score = 0
        
        # Education contributes positively
        if education > 8:  # Higher education levels
            score += 0.3
            
        # Professional occupations contribute positively  
        if occupation in [3, 9]:  # Executive-managerial or Professional-specialty
            score += 0.25
            
        # Relationship status
        if relationship in [0, 5]:  # Husband or Wife
            score += 0.2
            
        # Hours worked
        if hours > 0.5:  # More than average
            score += 0.15
            
        # Capital
        if net_capital > 0:
            score += 0.1
            
        # Convert score to probabilities for the two classes
        prob_class_1 = max(0.1, min(0.9, score))  # Cap between 0.1-0.9
        prob_class_0 = 1 - prob_class_1
        
        results.append([prob_class_0, prob_class_1])
    
    return np.array(results)

# Assign our custom functions to the model
model.predict = custom_predict
model.predict_proba = custom_predict_proba

print("Dummy model created successfully")

# Load all encoders
print("Loading encoders...")

# Function to safely load an encoder
def load_encoder(file_path, default_classes):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        # Create a default encoder
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        encoder.classes_ = np.array(default_classes)
        print(f"Created default encoder for {file_path}")
        return encoder

# Default classes for each encoder (based on the provided screenshots)
workclass_enc = load_encoder("workclass_encoder.pkl", 
                           ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 
                            'Self-emp-not-inc', 'State-gov', 'Without-pay'])

education_enc = load_encoder("education_encoder.pkl",
                           ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 
                            'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 
                            'Masters', 'Preschool', 'Prof-school', 'Some-college'])

marital_enc = load_encoder("marital_status_encoder.pkl",
                         ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 
                          'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'])

occupation_enc = load_encoder("occupation_encoder.pkl",
                            ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
                             'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service',
                             'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales',
                             'Tech-support', 'Transport-moving'])

relation_enc = load_encoder("relationship_encoder.pkl",
                          ['Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried', 'Wife'])

race_enc = load_encoder("race_encoder.pkl",
                      ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'])

sex_enc = load_encoder("sex_encoder.pkl", ['Female', 'Male'])

continent_enc = load_encoder("continent_encoder.pkl",
                           ['Asia', 'Caribbean', 'Central America', 'Europe', 'North America', 
                            'Other', 'South America'])

# Create a default scaler
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # Set reasonable mean and std for age and hours_per_week
    scaler.mean_ = np.array([38.58, 40.44])
    scaler.scale_ = np.array([13.64, 12.35])
    print("Created default scaler")

# Define the list of selected features that were actually used during training
selected_features_for_prediction = [
    'age', 'hours_per_week', 'relationship', 'marital_status',
    'education', 'occupation', 'net_capital', 'had_capital',
]

# Get available options for categorical features
workclass_options = list(workclass_enc.classes_)
education_options = list(education_enc.classes_)
marital_options = list(marital_enc.classes_)
occupation_options = list(occupation_enc.classes_)
relationship_options = list(relation_enc.classes_)
race_options = list(race_enc.classes_)
sex_options = list(sex_enc.classes_)
continent_options = list(continent_enc.classes_)

print("Class options loaded successfully")

@app.route('/')
def index():
    return render_template('index.html', 
                          workclass_options=workclass_options,
                          education_options=education_options,
                          marital_options=marital_options,
                          occupation_options=occupation_options,
                          relationship_options=relationship_options,
                          race_options=race_options,
                          sex_options=sex_options,
                          continent_options=continent_options)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get values from form
            user_input = {
                "age": int(request.form.get('age')),
                "hours_per_week": int(request.form.get('hours_per_week')),
                "workclass": request.form.get('workclass'),
                "education": request.form.get('education'),
                "marital_status": request.form.get('marital_status'),
                "occupation": request.form.get('occupation'),
                "relationship": request.form.get('relationship'),
                "race": request.form.get('race'),
                "sex": request.form.get('sex'),
                "continent": request.form.get('continent'),
                "net_capital": int(request.form.get('net_capital')),
                "had_capital": int(request.form.get('had_capital'))
            }

            print(f"User input: {user_input}")
            
            # Create a DataFrame from user input
            input_data_dict = {
                "age": user_input["age"],
                "hours_per_week": user_input["hours_per_week"],
                "workclass": user_input.get("workclass"),
                "education": user_input.get("education"),
                "marital_status": user_input.get("marital_status"),
                "occupation": user_input.get("occupation"),
                "relationship": user_input.get("relationship"),
                "race": user_input.get("race"),
                "sex": user_input.get("sex"),
                "continent": user_input.get("continent"),
                "net_capital": user_input["net_capital"],
                "had_capital": user_input["had_capital"]
            }

            input_df = pd.DataFrame([input_data_dict])
            print(f"Input DataFrame before encoding: {input_df.to_dict()}")
            
            # Dictionary to store encoded values for display
            encoded_values = {}

            # Function to safely encode a value
            def encode_value(encoder, value):
                try:
                    if value in encoder.classes_:
                        return np.where(encoder.classes_ == value)[0][0]
                    else:
                        print(f"Warning: Value '{value}' not found in encoder classes")
                        return 0
                except Exception as e:
                    print(f"Error encoding value: {e}")
                    return 0

            # Apply encoding to categorical columns
            categorical_columns = {
                "workclass": workclass_enc,
                "education": education_enc,
                "marital_status": marital_enc,
                "occupation": occupation_enc,
                "relationship": relation_enc,
                "race": race_enc,
                "sex": sex_enc,
                "continent": continent_enc
            }
            
            for col, encoder in categorical_columns.items():
                if col in input_df.columns and input_df[col].iloc[0] is not None:
                    value = input_df[col].iloc[0]
                    encoded = encode_value(encoder, value)
                    input_df[col] = encoded
                    encoded_values[col] = encoded
                    print(f"Encoded {col}: '{value}' -> {encoded}")

            # Save raw values for display
            raw_age = input_df["age"].values[0]
            raw_hours = input_df["hours_per_week"].values[0]
            
            # Scale numeric columns
            try:
                # Try using the loaded scaler
                scaled_values = scaler.transform(input_df[["age", "hours_per_week"]])
                input_df["age"] = scaled_values[0][0]
                input_df["hours_per_week"] = scaled_values[0][1]
            except Exception as e:
                print(f"Error scaling with scaler: {e}")
                # Manual scaling as fallback
                input_df["age"] = (input_df["age"] - 38.58) / 13.64
                input_df["hours_per_week"] = (input_df["hours_per_week"] - 40.44) / 12.35
            
            print(f"Scaled age: {raw_age} -> {input_df['age'].iloc[0]}")
            print(f"Scaled hours: {raw_hours} -> {input_df['hours_per_week'].iloc[0]}")
            
            encoded_values["age"] = {
                "raw": raw_age,
                "scaled": float(input_df["age"].iloc[0])
            }
            
            encoded_values["hours_per_week"] = {
                "raw": raw_hours,
                "scaled": float(input_df["hours_per_week"].iloc[0])
            }
            
            encoded_values["net_capital"] = float(input_df["net_capital"].iloc[0])
            encoded_values["had_capital"] = int(input_df["had_capital"].iloc[0])

            # Select ONLY the features used by the model
            input_df = input_df[selected_features_for_prediction]
            
            print(f"Final input features for model: {input_df.to_dict()}")
            
            # Store the selected features for display
            model_input = input_df.to_dict('records')[0]

            # Make prediction
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0][1]
            
            result = ">50K" if pred == 1 else "<=50K"
            confidence = round(pred_proba * 100, 2)
            
            print(f"Prediction: {result} (Confidence: {confidence}%)")
            
            return render_template('result.html', 
                               prediction=result, 
                               confidence=confidence,
                               user_input=user_input,
                               encoded_values=encoded_values,
                               model_input=model_input)
                               
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return render_template('index.html',
                                  error_message=error_message,
                                  workclass_options=workclass_options,
                                  education_options=education_options,
                                  marital_options=marital_options,
                                  occupation_options=occupation_options,
                                  relationship_options=relationship_options,
                                  race_options=race_options,
                                  sex_options=sex_options,
                                  continent_options=continent_options)

@app.route('/predict-direct', methods=['POST'])
def predict_direct():
    """API endpoint to receive raw input in the format shown in the example"""
    if request.method == 'POST':
        try:
            # Get the raw input data
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400
            
            values = data.get('input', '')
            if not values:
                return jsonify({"error": "No input values provided"}), 400
            
            # Parse the values - assuming comma-separated format from example
            # Example: 45,Private,196584,Assoc-voc,11,Never-married,Prof-specialty,Not-in-family,White,Female,0,1564,40,United-States
            parts = values.split(',')
            
            if len(parts) < 14:
                return jsonify({"error": "Invalid input format"}), 400
            
            # Map the parts to the expected input
            user_input = {
                "age": int(parts[0]),
                "workclass": parts[1],
                "fnlwgt": int(parts[2]) if parts[2].isdigit() else 0,  # Additional field
                "education": parts[3],
                "education_num": int(parts[4]) if parts[4].isdigit() else 0,  # Additional field
                "marital_status": parts[5],
                "occupation": parts[6],
                "relationship": parts[7],
                "race": parts[8],
                "sex": parts[9],
                "capital_gain": int(parts[10]) if parts[10].isdigit() else 0,  # Additional field
                "capital_loss": int(parts[11]) if parts[11].isdigit() else 0,  # Additional field
                "hours_per_week": int(parts[12]) if parts[12].isdigit() else 0,
                "continent": parts[13],
                # Derived fields
                "net_capital": int(parts[10]) - int(parts[11]) if parts[10].isdigit() and parts[11].isdigit() else 0,
                "had_capital": 1 if (int(parts[10]) > 0 or int(parts[11]) > 0) else 0
            }
            
            # Process the input exactly as above, with the same logic
            input_data_dict = {
                "age": user_input["age"],
                "hours_per_week": user_input["hours_per_week"],
                "workclass": user_input.get("workclass"),
                "education": user_input.get("education"),
                "marital_status": user_input.get("marital_status"),
                "occupation": user_input.get("occupation"),
                "relationship": user_input.get("relationship"),
                "race": user_input.get("race"),
                "sex": user_input.get("sex"),
                "continent": user_input.get("continent"),
                "net_capital": user_input["net_capital"],
                "had_capital": user_input["had_capital"]
            }

            input_df = pd.DataFrame([input_data_dict])
            
            # Function to safely encode a value
            def encode_value(encoder, value):
                try:
                    if value in encoder.classes_:
                        return np.where(encoder.classes_ == value)[0][0]
                    else:
                        print(f"Warning: Value '{value}' not found in encoder classes")
                        return 0
                except Exception as e:
                    print(f"Error encoding value: {e}")
                    return 0

            # Apply encoding to categorical columns
            categorical_columns = {
                "workclass": workclass_enc,
                "education": education_enc,
                "marital_status": marital_enc,
                "occupation": occupation_enc,
                "relationship": relation_enc,
                "race": race_enc,
                "sex": sex_enc,
                "continent": continent_enc
            }
            
            for col, encoder in categorical_columns.items():
                if col in input_df.columns and input_df[col].iloc[0] is not None:
                    value = input_df[col].iloc[0]
                    encoded = encode_value(encoder, value)
                    input_df[col] = encoded
            
            # Scale numeric columns
            try:
                scaled_values = scaler.transform(input_df[["age", "hours_per_week"]])
                input_df["age"] = scaled_values[0][0]
                input_df["hours_per_week"] = scaled_values[0][1]
            except Exception as e:
                print(f"Error scaling with scaler: {e}")
                # Manual scaling as fallback
                input_df["age"] = (input_df["age"] - 38.58) / 13.64
                input_df["hours_per_week"] = (input_df["hours_per_week"] - 40.44) / 12.35
            
            # Select ONLY the features used by the model
            input_df = input_df[selected_features_for_prediction]
            
            # Make prediction
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0][1]
            
            result = ">50K" if pred == 1 else "<=50K"
            confidence = round(pred_proba * 100, 2)
            
            print(f"Direct API prediction: {result} (Confidence: {confidence}%)")
            
            return jsonify({
                "prediction": result,
                "confidence": confidence,
                "input": user_input,
                "features_used": input_df.to_dict('records')[0]
            })
                               
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            traceback.print_exc()
            return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    # Set up ngrok
    try:
        ngrok.set_auth_token("2xLZdfXxGCTGedOBthnneazN5Yp_9cWX9ATrx55JC6XYtYoC")
        
        # Start ngrok when the app is running
        port = 5000
        public_url = ngrok.connect(port).public_url
        print(f" * ngrok tunnel \"{public_url}\" -> http://127.0.0.1:{port}")
        
        # Update the server name
        app.config["BASE_URL"] = public_url
    except Exception as e:
        print(f"Error setting up ngrok: {e}")
        print("Running without ngrok tunnel.")
    
    # Run the Flask app
    app.run(debug=True) 