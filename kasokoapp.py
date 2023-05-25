from flask import Flask,request,jsonify,render_template, redirect, url_for
import numpy as np
import sklearn
import pickle
from sklearn import linear_model
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)
models=pickle.load(open('ModelKasoko.pkl','rb'))
@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
	models=pickle.load(open('ModelKasoko.pkl','rb'))
	int_futures=[float(i) for i in request.form.values()]
	dernier_futures=[np.array(int_futures)]
	dernier_futures=np.array([dernier_futures]).reshape(1,8)
	Predire=models.predict(dernier_futures)
	if (models.predict(dernier_futures)==0):
		Predire="Negatif"
	else:
		Predire="Positif"

	return render_template('index.html',prediction_text_="Votre diagnostic est:{}".format(Predire))

if __name__=='__main__':
	app.run(debug=True)
	