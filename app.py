import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import math

app = Flask(__name__,template_folder="template", static_folder="staticFile")
model = pickle.load(open('RF_Model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features  = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = np.round(prediction[0],4)
    return render_template('index.html',prediction_text="covid death rate for upcoming week will be  {}".format(math.floor(output)))


if __name__ == "__main__":
    app.run(debug=True)