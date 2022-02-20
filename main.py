from flask import Flask, render_template, request, jsonify
import nltk
import matplotlib
import joblib
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
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
    # read image yang udh diambil dari canvas
    preProcessedText = preprocessed(text)
    tfidfconverter = joblib.load("./tfidf-train.joblib")
    text_tfidf = tfidfconverter.transform([preProcessedText])

    outRf = modelRf.predict(text_tfidf)[0]
    outSVM = modelSVM.predict(text_tfidf)[0]
    label = ["Negative", "Neutral", "Positive"]
    print(label[outRf-1])
    return "<h1 id='res'> Random Forest Classifier: " + str(label[outRf-1]) + "</h1>" + "<h1 id='res'> SVM Classifier: " + str(label[outSVM-1]) + "</h1>"

def stopWordAndStem(inputStr):
    listStopword = set(stopwords.words('english'))
    lemma = nltk.wordnet.WordNetLemmatizer()
    tokens = word_tokenize(inputStr)
    removed = []
    for i in tokens:
        if i not in listStopword:
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

    # Remove all mention
    result = re.sub("@[A-Za-z0-9_]+", "", result)

    # Remove all hashtag
    result = re.sub("#[A-Za-z0-9_]+", "", result)

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

if __name__ == '__main__':
    app.run(debug=True)
