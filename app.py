import numpy as np
from flask import Flask, request, render_template
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model from the .pkl file
model = pickle.load(open('model.pkl', 'rb'))

# Define the route for the home page, which displays the form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route that handles the form submission and makes a prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form as a list of strings
    form_values = [x for x in request.form.values()]
    
    # Convert the string inputs to a list of floating-point numbers
    float_features = [float(x) for x in form_values]
    
    # Create a NumPy array with the correct shape for the model
    final_features = [np.array(float_features)]
    
    # Use the model to make a prediction
    prediction = model.predict(final_features)

    # Get the single prediction value and round it to two decimal places
    output = round(prediction[0][0], 2)

    # Render the HTML page again, this time including the prediction result
    return render_template('index.html', prediction_text=f'Predicted Performance Index: {output}')

# This block runs the app when the script is executed directly
if __name__ == "__main__":
    app.run(debug=True)