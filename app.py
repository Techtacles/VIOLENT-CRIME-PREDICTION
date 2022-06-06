#Importing the necessary tools
from flask import Flask,request, jsonify, render_template
import numpy as np
import pickle

import pickle

#Loading the saved models in the ipynb file
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
ohe=pickle.load(open('ohe_encoder.pkl','rb'))
sc=pickle.load(open('sc.pkl','rb'))
label_encoder=pickle.load(open('label_encoder.pkl','rb'))

#Creating url routing for the flask web app
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    year=request.form['year']
    month=request.form['month']
    day=request.form['day']
    area=request.form['area']
    categorical_features=ohe.transform([[month.lower(),area.lower()]]).toarray()
    numeric_features=sc.transform([[year,day]])
    joined=np.hstack([categorical_features,numeric_features])
    prediction=model.predict(joined)
    transformed_prediction=label_encoder.inverse_transform(prediction)
    return render_template('result.html',prediction_text=f'{transformed_prediction[0]}',store_month=month,store_year=year,store_area=area)
        
    
@app.route('/contact')
def contact():
    return render_template("contact.html")


if __name__ == '__main__':
   app.run(debug=False)