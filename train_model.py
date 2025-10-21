import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def train_and_save_model():
    df = pd.read_csv("IMDB Dataset.csv")

    X = df["review"]
    y = df["sentiment"]

    model = Pipeline([
     ("tfidf", TfidfVectorizer(stop_words="english")),
     ("nb", MultinomialNB())
])

    model.fit(X,y)

    joblib.dump(model, "model.joblib")
if __name__ == "__main__":
    train_and_save_model()
