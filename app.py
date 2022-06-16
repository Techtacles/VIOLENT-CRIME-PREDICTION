#Importing the necessary tools
from flask import Flask,request, jsonify, render_template,redirect,url_for
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle


import os

basedir = os.path.abspath(os.path.dirname(__file__))

#Loading the saved models in the ipynb file
app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
ohe=pickle.load(open('ohe_encoder.pkl','rb'))
sc=pickle.load(open('sc.pkl','rb'))
label_encoder=pickle.load(open('label_encoder.pkl','rb'))

#Creating url routing for the flask web app
app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'database.db')
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['UPLOAD_FOLDER'] = 'static/'


db = SQLAlchemy(app)


####MODELSSS

class Crimes(db.Model):
    __tablename__ = 'crimes '
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.Integer, unique=False, nullable=False)
    month = db.Column(db.Integer, unique=False, nullable=False)
    year = db.Column(db.Integer, unique=False, nullable=False)
    area = db.Column(db.String(150), unique=False, nullable=False)
    crime = db.Column(db.String(150), unique=False, nullable=False)
    casualties=db.Column(db.Integer, unique=False, nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.current_timestamp())

    def __repr__(self):
        return f"User('{self.day}', '{self.month}','{self.year}')"



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
        
    
@app.route('/report_crime' ,methods=['POST','GET'])
def contact():
    if request.method == "POST":
        form = request.form
        day = request.form['day']
        month = request.form['month']
        year = request.form['year']
        crime = request.form['crime']
        location = request.form['location']
        casualty=request.form['casualty']
        data = Crimes(day=day, month=month, year = year, crime = crime, area = location,casualties=casualty)
        db.session.add(data)
        db.session.commit()

        return redirect(url_for('reports'))
    return render_template("contact.html")


@app.route('/reports' ,methods=['POST','GET'])
def reports():
    reports = Crimes.query.all()
    return render_template("reports.html", reports = reports)

if __name__ == '__main__':
   app.run(debug=True)