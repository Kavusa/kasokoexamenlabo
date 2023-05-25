from flask import FLASK,request,jsonify,render_template, redirect, url_for
import numpy as np
import sklearn
import pickle
from sklearn.ensemble import RandomForestRegressor

app=Flask(__name__)
models=pickle.load(open('ModelKasoko.pkl','rb'))
dict_classe_lesion={
0:"Negatif",
1:"Positif"
}

@app.route('/')
def home():
	return render_template('index.html')

from distutils.log import debug	
@app.route('/predict',methods=['POST'])

def predict():
	models=pickle.load(open('ModelKasoko.pkl','rb'))
	int_futures=[float(i) for i in request.form.values()]
	dernier_futures=[np.array(int_futures)]
	dernier_futures=np.array([dernier_futures]).reshape(1,8)
	Predire=models.predict(dernier_futures)
	Pred_class=Predire.argmax(axis=-1)
	Prediction=dict_classe_lesion[Predire[0]]
	Result=str(Prediction)
	return render_template('index.html',prediction_text_="Votre type est:{}".format(Result))

if __name__=='__main__':
	app.run(debug=True)
	