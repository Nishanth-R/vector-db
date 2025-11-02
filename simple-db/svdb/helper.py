# Lazy loading of stopwords to avoid slow startup
_stop_words = None

def _get_stop_words():
    """Lazily load stopwords only when needed."""
    global _stop_words
    if _stop_words is None:
        try:
            import nltk
            from nltk.corpus import stopwords
            try:
                stopwords.words('english')
            except LookupError:
                nltk.download('stopwords')
            _stop_words = set(stopwords.words('english'))
        except Exception:
            # Fallback to empty set if NLTK fails
            _stop_words = set()
    return _stop_words

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
    stop_words = _get_stop_words()
    return " ".join(word for word in text.split() if word not in stop_words)
