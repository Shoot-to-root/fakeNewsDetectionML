from flask import Flask, render_template, request
from sklearn.externals import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    try:
        if request.method == 'GET':
            return render_template('home.html')

    except: 
        return render_template('404.html')

@app.route('/home', methods=['GET', 'POST'])
def home_page():
    try:
        if request.method == 'GET': 
            return render_template('home.html')
    except:
        return render_template('404.html')

@app.route('/home/result', methods=['GET'])
def result():
    try:
        if request.method == 'GET':
            return render_template('result.html') 
    except:
        return render_template('404.html')

@app.route('/demo', methods=['GET'])
def demo_page():
    try:
        if request.method == 'GET':
            return render_template('demo.html')

    except:
        return render_template('404.html')

@app.route('/article_result', methods=['POST'])
def article_result():
	if request.method == 'POST':
		text = request.form['text']
		corpus = []
		text = re.sub('[^a-zA-Z]', ' ', text)
		text = text.lower()
		text = text.split()
		lemmatizer = WordNetLemmatizer()
		text = [lemmatizer.lemmatize(word) for word in text if not word in set(
			stopwords.words('english'))]
		text = ' '.join(text)
		corpus.append(text)
		tfidf = joblib.load('vectorizer.pkl')
		tfidf_corpus = tfidf.transform(corpus)
		classifier = joblib.load('classifier.pkl')
		prob = classifier.predict_proba(tfidf_corpus)[:, 1]
		prob = prob*100
		prob = "%.2f" % prob
		this = float(prob)
		return render_template('result.html', value=prob, this=this)
        
