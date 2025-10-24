# config.py
"""Configuration for paths, hyperparameters, and constants."""

DATA_DIR = "./data/"
TRAIN_FILE = r"D:\Acadamics\GitHub\NLP\native_language_identification\data\Stansford_english_corpus_sentences_train.csv"   # placeholder for dataset
TEST_FILE = r'D:\Acadamics\GitHub\NLP\native_language_identification\data\Stansford_english_corpus_sentences_test.csv'     # placeholder for dataset
MODEL_DIR = "models/"
RESULTS_DIR = "results/"

# Kernel and model parameters
NGRAM_MIN = 1
NGRAM_MAX = 1
SVM_C = 1.0
SVM_KERNEL = "precomputed"
SVM_RANDOM_STATE = 42
LOWERCASE = True
REMOVE_PUNCT = True
REMOVE_DIGITS = True
REMOVE_STOPWORDS = True
MIN_TOKEN_LEN = 2