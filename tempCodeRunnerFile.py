from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class PredictionModel:
    def __init__(self, stopwords_file, vectorizer_file, model_file):
        self.stopwords = self._load_stopwords(stopwords_file)
        self.vectorizer = self._initialize_vectorizer(vectorizer_file)
        self.model = self._load_model(model_file)

    def _load_stopwords(self, file_path):
        with open(file_path, "r") as file:
            return file.read().splitlines()

    def _initialize_vectorizer(self, file_path):
        vocabulary = pickle.load(open(file_path, "rb"))
        return TfidfVectorizer(stop_words=self.stopwords, lowercase=True, vocabulary=vocabulary)

    def _load_model(self, file_path):
        return pickle.load(open(file_path, 'rb'))

    def predict(self, text):
        transformed_input = self.vectorizer.fit_transform([text])
        return self.model.predict(transformed_input)[0]

class WebApp:
    def __init__(self, prediction_model):
        self.app = Flask(__name__)
        self.prediction_model = prediction_model
        self._add_routes()

    def _add_routes(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            prediction = None
            if request.method == 'POST':
                user_input = request.form['text']
                prediction = self.prediction_model.predict(user_input)
            return render_template('index.html', prediction=prediction)

    def run(self, debug=False):
        self.app.run(debug=debug)

if __name__ == '__main__':
    model = PredictionModel(stopwords_file="stopwords.txt", 
                            vectorizer_file="tfidfvectoizer.pkl", 
                            model_file="LinearSVCTuned.pkl")
    app = WebApp(prediction_model=model)
    app.run(debug=True)
