# preprocessing.py
"""Text preprocessing utilities: cleaning, tokenization, n-gram extraction."""

import re
from typing import List, Iterable
from config import LOWERCASE, REMOVE_PUNCT, REMOVE_DIGITS, REMOVE_STOPWORDS, MIN_TOKEN_LEN
import nltk
from nltk.tokenize import word_tokenize
from nltk import ngrams
from collections import Counter
from bs4 import BeautifulSoup
import re
import string

# Ensure punkt is available (user must run once or we handle gracefully)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

if REMOVE_STOPWORDS:
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
else:
    STOPWORDS = set()



def remove_ns_tag(text):
  l1 = text.replace('"',"")
  start = 0
  end = len(l1)
  i = 0
  count  = l1.count("<NS")
  while i <= count:
    last_index_ns = l1.find("<NS", start, end)
    last_index_c = l1.find("<c",last_index_ns,end)
    re = l1[last_index_ns:last_index_c]
    l1 = l1.replace(re,"")
  
    i = i+1

  return l1

def create_space(text):
  return text.replace("<"," <")



# function to remove HTML tags
def remove_html_tags(text):
    result = BeautifulSoup(text, 'html.parser').get_text()
    return result


def remove_additional_spaces(text):
  return re.sub(" +"," ",text)


def remove_slash_n(text):
  text = text.replace("\n","")
  text = remove_additional_spaces(text)
  #text = remove_some_words(text)
  return text



def remove_some_words1(text):
  some_words = ["dear","sir","madam","mrs","mr","ms"]
  querywords = text.split(" ")

  resultwords  = [word for word in querywords if word.lower() not in some_words]
  result = ' '.join(resultwords)

  return result



def get_ngrams(text, n ):
    n_grams = ngrams(word_tokenize(text), n)
    return [' '.join(grams) for grams in n_grams]
















import re
import string

def preprocess_data_frames(train_df, test_df, lowercase=True):
    """
    Preprocess train and test DataFrames in a vectorized and efficient way.
    
    Steps:
    - Remove ns tags
    - Create spaces if needed
    - Remove HTML tags
    - Remove extra spaces and newline characters
    - Lowercase text and labels (optional)
    - Strip leading/trailing spaces
    - Remove punctuation
    - Remove some specific words
    """

    def preprocess_text_series(series):
        series = series.apply(remove_ns_tag)
        series = series.apply(create_space)
        series = series.apply(remove_html_tags)
        series = series.apply(remove_additional_spaces)
        series = series.apply(remove_slash_n)
        if lowercase:
            series = series.str.lower()
        series = series.str.strip()
        # Remove punctuation
        series = series.apply(lambda x: re.sub(f"[{re.escape(string.punctuation)}]", "", x))
        # Remove specific words
        series = series.apply(remove_some_words1)
        return series

    # Preprocess text
    train_df["text"] = preprocess_text_series(train_df["text"])
    test_df["text"] = preprocess_text_series(test_df["text"])

    # Preprocess labels
    if lowercase:
        train_df["native_language"] = train_df["native_language"].str.lower()
        test_df["native_language"] = test_df["native_language"].str.lower()

    return train_df, test_df
