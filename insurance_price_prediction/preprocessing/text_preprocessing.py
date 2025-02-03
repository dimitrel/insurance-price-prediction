import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(product_name, keep_numbers=False):

    """Remove non-alphabetic characters and collapse multiple spaces."""

    text = str(product_name).strip().lower()

    if not keep_numbers:
        cleaned_text = re.sub(r'[^a-z]+', ' ', text).strip()
    else:
        cleaned_text = re.sub(r'[^a-z0-9]+', ' ', text).strip()
    return cleaned_text


def preprocess_text_with_nltk(text :str, lemmatizer=True) -> str:

    """ Processing of text data leveraging nltk functionality removing stopwords and applying lemmatizer."""

    text = clean_text(text)

    # text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)

    # Removing Stop Words
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    if lemmatizer:
        lemmatizer = WordNetLemmatizer()
        lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return " ".join(lemmatized_tokens)
