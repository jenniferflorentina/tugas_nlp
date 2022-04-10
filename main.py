from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import matplotlib
import joblib
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

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
    modelSVMCBOW = joblib.load("./svmCBOW.joblib")
    modelSVMSG = joblib.load("./svmSG.joblib")

    # read image yang udh diambil dari canvas
    preProcessedText = preprocessed(text)
    tfidfconverter = joblib.load("./tfidf-train.joblib")
    text_tfidf = tfidfconverter.transform([preProcessedText])

    outRf = modelRf.predict(text_tfidf)[0]
    outSVM = modelSVM.predict(text_tfidf)[0]
    outSVMLinear = modelSVMLinear.predict(text_tfidf)[0]

    stemPreprocessed = word_tokenize(preProcessedText)
    cbowConverter = Word2Vec.load("trainCBOW.model")
    text_cbow = getVectors([stemPreprocessed],cbowConverter)
    sgConverter = Word2Vec.load("trainSkipGram.model")
    text_sg = getVectors([stemPreprocessed],sgConverter)

    outSVMCBOW = modelSVMCBOW.predict(text_cbow)[0]
    outSVMSG = modelSVMSG.predict(text_sg)[0]

    label = ["Negative", "Neutral", "Positive"]
    print(label[outRf-1])
    return "<h2 id='res'> TFIDF </h2>" \
           "<ul> <li><h3>Random Forest Classifier: " + str(label[outRf-1]) + "</h3></li>" \
           + "<li><h3> SVM Classifier: " + str(label[outSVM-1]) + "</h3></li>" + \
           "<li><h3> SVM Linear Classifier: " + str(label[outSVMLinear-1]) + "</h3></li> </ul> <br>" \
           "<h2 id='res'> Word2Vec </h2>"\
           "<ul> <li><h3>SVM CBOW Classifier: " + str(label[outSVMCBOW-1]) + "</h3></li>" \
           "<li><h3> SVM Skip Gram Classifier: " + str(label[outSVMSG-1]) + "</h3></li> <ul>"


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

embeddingsSize=256
def getVectors(dataset, model):
    singleDataItemEmbedding=np.zeros(embeddingsSize)
    vectors=[]
    for dataItem in dataset:
        wordCount=0
        for word in dataItem:
            if word in model.wv.key_to_index.keys():
                singleDataItemEmbedding=singleDataItemEmbedding+model.wv[word]
                wordCount=wordCount+1
        if wordCount > 0:
            singleDataItemEmbedding= singleDataItemEmbedding/wordCount
        vectors.append(singleDataItemEmbedding)
    return vectors

if __name__ == '__main__':
    app.run(debug=True)
