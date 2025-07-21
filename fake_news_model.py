import pickle
import re

# Load saved model
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load saved vectorizer
with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Function to preprocess text
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text.strip()

# Function to predict fake news
def predict_fake_news(news):
    cleaned_news = clean_text(news)
    vectorized_news = vectorizer.transform([cleaned_news])
    prediction = model.predict(vectorized_news)
    return "Fake News" if prediction[0] == 1 else "Real News"



