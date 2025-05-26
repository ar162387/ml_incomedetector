# Income Prediction Web Application

This is a Flask web application that serves a machine learning model to predict income levels based on various socio-economic factors.

## Features
- Web-based interface for model predictions
- Form with dropdown menus and input fields for user data
- Visual result display with confidence percentage
- Accessible via ngrok tunneling

## Setup and Installation

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

1. Make sure all model and encoder files (.pkl) are in the root directory.
2. Run the Flask application:
   ```
   python app.py
   ```
3. The application will start running on http://127.0.0.1:5000/
4. An ngrok tunnel will be automatically created, and the public URL will be displayed in the console.

## Files Description
- `app.py`: Main Flask application
- `templates/index.html`: Input form for user data
- `templates/result.html`: Results display page
- `requirements.txt`: Python dependencies
- `.pkl files`: Pre-trained model and encoders

## Model Information
The model predicts whether an individual's income is above or below $50K based on:
- Age
- Hours per week
- Relationship status
- Marital status
- Education
- Occupation
- Net capital
- Had capital indicator

## Technologies Used
- Flask (Web framework)
- Bootstrap (Frontend styling)
- Pandas & NumPy (Data processing)
- Scikit-learn (ML model)
- Ngrok (Tunneling service) 