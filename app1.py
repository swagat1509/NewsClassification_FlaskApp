from distutils.debug import DEBUG
from operator import index
from pickle import TRUE
from flask import Flask, render_template, request
import os, joblib
import sklearn


## Vectorizer
def load_model(filepath):
    return joblib.load(open(os.path.join(filepath),"rb"))


def get_label(key,dicti):
    return dicti[key]



app = Flask(__name__)

@app.route('/')
def basic():
    return render_template('index.html')


@app.route('/predict1',methods=['GET','POST'])
def predict1():
    if request.method == 'POST':
        txt = request.form['rawtext']
        return render_template('index.html',t = txt+"new thing")


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        txt = request.form['rawtext']
        vectorized_text = load_model('static/models/final_news_cv_vectorizer.pkl').transform([txt]).toarray() 
        modelchoice = request.form['model_choice']

        if modelchoice == "nb":
            news_classifier_model = load_model('static/models/newsclassifier_NB_model.pkl')

        if modelchoice == "logit":
            news_classifier_model = load_model('static/models/newsclassifier_Logit_model.pkl')


        if modelchoice == "rf":
            news_classifier_model = load_model('static/models/newsclassifier_RFOREST_model.pkl')

        prediction_lebels = {0: 'business',1:'tech',2:'health',3:'politics',4:'entertainment'}
        prediction = int(news_classifier_model.predict(vectorized_text))
        result = get_label(prediction,prediction_lebels)




        return render_template('index.html',t = txt.upper(), r = result)




if __name__ == '__main__':
    app.run(debug = True)