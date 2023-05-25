from flask import Flask
import pickle
import pandas as pd
from sklearn import linear_model
from joblib import dump, load
 
app = Flask(__name__)
models=pickle.load(open('ModelKasoko2.pkl','rb'))
dict_classe_lesion={
1:"Positif",
0:"Negatif"
}

@app.route('/')
def home():
	return render_template('index.html')

from distutils.log import debug	
 
@app.route('/predict/<prediction>')
def predict():
	models=pickle.load(open('ModelKasoko2.pkl','rb'))
	int_futures=[float(i) for i in request.form.values()]
	dernier_futures=[np.array(int_futures)]
	dernier_futures=np.array([dernier_futures]).reshape(1,8)
	Predire=models.predict(dernier_futures)
	Pred_class=Predire.argmax(axis=-1)
	Prediction=dict_classe_lesion[Predire[0]]
	Result=str(Prediction)
	return render_template('index.html',prediction_text_="Votre diagnostic est:{}".format(Result))

if __name__ == '__main__':
      app.run(debug=True)
	
import pandas as pd
import pickle
from sklearn import linear_model
from joblib import dump, load
 
app = Flask(__name__)
models=pickle.load(open('ModelKasoko2.pkl','rb'))
dict_classe_lesion={
1:"Positif",
0:"Negatif"
}
 
@app.route('/predict/<prediction>')
def home():
	return render_template('index.html')

def predict(prediction):
    regr = load('ModelKasoko2.pkl')
    return str(regr.predict([[int(prediction)]]))
 
if __name__ == '__main__':
      app.run(debug=True)
