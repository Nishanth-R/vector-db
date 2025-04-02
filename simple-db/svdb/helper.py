import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def filter_stopwords_in_text(text):
    """
    Filter stopwords from text and return the filtered text as a string.
    
    Args:
        text (str): Input text to filter
        
    Returns:
        str: Filtered text with stopwords removed
    """
    if not isinstance(text, str):
        return ""
    return " ".join(word for word in text.split() if word not in stop_words)
