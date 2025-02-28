import nltk
from nltk.corpus import stopwords
from database import Database
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

db = Database()
bow = db.load_bow()

encoded_stop_words = [bow[word] for word in stop_words if word in bow]

def filter_stopwords_in_text(text):
    new_words = []
    for word in text.split():
        if word not in stop_words:
            new_words.append(word)
    return new_words

def filter_stopwords_encoded(text_encoded):
    new_words = []
    for word in text_encoded:
        if word in encoded_stop_words:
            new_words.append(word)
    return new_words
