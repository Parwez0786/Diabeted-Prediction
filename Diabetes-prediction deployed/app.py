from flask import Flask, render_template, request
import pickle
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier  # Import the correct class
from sklearn.tree import DecisionTreeClassifier  # Explicitly import DecisionTreeClassifier

# Fix the old path issues by mapping the old module to the new one
sys.modules['sklearn.ensemble.forest'] = sys.modules['sklearn.ensemble._forest']
sys.modules['sklearn.tree.tree'] = sys.modules['sklearn.tree._tree']

# Define a custom unpickler
class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DecisionTreeClassifier':
            return DecisionTreeClassifier
        return super().find_class(module, name)

# Load the model with custom unpickler
filename = 'diabetes-prediction-rfc-model.pkl'
with open(filename, 'rb') as file:
    classifier = CustomUnpickler(file).load()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Collect form data
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        # Prepare data for prediction
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])

        # Predict using the loaded classifier
        my_prediction = classifier.predict(data)

        # Return the result to the result.html template
        return render_template('result.html', prediction=my_prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
