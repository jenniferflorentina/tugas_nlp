from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import matplotlib
import joblib
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

matplotlib.use('Agg')

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # get data from textarea
    json = request.json
    text = json.get('inputUser')
    # load model yang udh di training
    modelRf = joblib.load("./rf.joblib")
    modelSVM = joblib.load("./svm.joblib")
    modelSVMLinear = joblib.load("./svmLinear.joblib")
    modelLSTM= keras.models.load_model('./lstm_tfidf.h5')
    modelSVMCBOW = joblib.load("./svmCBOW.joblib")
    modelLSTMCBOW= keras.models.load_model('./lstm.h5')

    # read image yang udh diambil dari canvas
    preProcessedText = preprocessed(text)

    #TFIDF
    tfidfconverter = joblib.load("./tfidf-train.joblib")
    text_tfidf = tfidfconverter.transform([preProcessedText])

    outRf = modelRf.predict(text_tfidf)[0]
    outSVM = modelSVM.predict(text_tfidf)[0]
    outSVMLinear = modelSVMLinear.predict(text_tfidf)[0]

    #Word2Vec
    cbowConverter = Word2Vec.load("trainCBOW.model")
    text_cbow = buildWordVector(preProcessedText, embeddingsSize, cbowConverter)
    outSVMCBOW = modelSVMCBOW.predict(text_cbow)[0]

    #LSTM
    tok = joblib.load("./tokenizer.joblib")
    encd_rev = tok.texts_to_sequences([preProcessedText])
    padTextCBOW = pad_sequences(encd_rev, maxlen=24, padding='post')
    # TFIDF
    outLSTM = modelLSTM.predict(padTextCBOW)[0].argmax()
    print(outLSTM)
    # Word2Vec
    outLSTMCBOW = modelLSTMCBOW.predict(padTextCBOW)[0].argmax()
    print(outLSTMCBOW)

    label = ["Negative", "Neutral", "Positive"]
    return "<h2 id='res'> TFIDF </h2>" \
           "<ul> <li><h3>Random Forest Classifier: " + str(label[outRf]) + "</h3></li>" \
           + "<li><h3> SVM Classifier: " + str(label[outSVM]) + "</h3></li>" + \
           "<li><h3> SVM Linear Classifier: " + str(label[outSVMLinear]) + "</h3></li>"+\
           "<li><h3> Bidirectional LSTM: " + str(label[outLSTM]) + "</h3></li> </ul> <br>" \
           "<h2 id='res'> Word2Vec CBOW </h2>"\
           "<ul> <li><h3>SVM CBOW Classifier: " + str(label[outSVMCBOW]) + "</h3></li>" \
           "<li><h3> Bidirectional LSTM: " + str(label[outLSTMCBOW]) + "</h3></li> <ul>"


def stopWordAndStem(inputStr):
    lemma = nltk.wordnet.WordNetLemmatizer()
    tokens = word_tokenize(inputStr)
    removed = []
    for i in tokens:
        removed.append(lemma.lemmatize(i))
    return " ".join(removed)

def preprocessed(sentence):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

    s = sentence.replace('\n', '')

    s = re.sub(r"http\S+", "", s)

    result = s.replace("&amp;", "");

    # Remove all emoji
    result = emoji_pattern.sub(r'', result)

    # Remove all the special characters
    result = re.sub(r'\W', ' ', result)

    # remove all single characters
    result = re.sub(r'\s+[a-zA-Z]\s+', ' ', result)

    # Substituting multiple spaces with single space
    result = re.sub(r'\s+', ' ', result, flags=re.I)

    result = re.sub(r"\d+", "", result)

    result = result.strip()

    # Converting to Lowercase
    result = result.lower()

    return stopWordAndStem(result)

embeddingsSize=300
def buildWordVector(text, size, model):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

if __name__ == '__main__':
    app.run(debug=True)
