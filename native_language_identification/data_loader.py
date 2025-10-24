# data_loader.py
"""Data loading utilities.
Loads CSVs, detects text/label columns, returns pandas DataFrames or lists.
"""

from pathlib import Path
import pandas as pd
from typing import Tuple, Optional
from config import TRAIN_FILE, TEST_FILE

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file from path (Path or str)."""
    return pd.read_csv(path)

def detect_text_label_columns(df: pd.DataFrame,
                              prefer_text_cols=("text", "essay", "content"),
                              prefer_label_cols=("label", "lang", "language", "target")):
    """Try to find sensible text and label columns in a DataFrame."""
    cols = list(df.columns)
    text_col = None
    label_col = None
    lc = [c.lower() for c in cols]
    for pref in prefer_text_cols:
        for i, c in enumerate(lc):
            if pref in c:
                text_col = cols[i]
                break
        if text_col:
            break
    for pref in prefer_label_cols:
        for i, c in enumerate(lc):
            if pref in c:
                label_col = cols[i]
                break
        if label_col:
            break
    return text_col, label_col

# def load_csv(train_path: Optional[str] = None,
#               test_path: Optional[str] = None):
#     """Load train and test DataFrames. If None, uses config placeholders."""
#     train_path = train_path or TRAIN_FILE
#     test_path = test_path or TEST_FILE
#     train_df = load_csv(train_path)
#     test_df = load_csv(test_path)
#     return train_df, test_df


def subset_data(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame):
    train_data = train_dataframe[["sent","ans1","ans2","native_language"]]
    test_data = test_dataframe[["sent","ans1","ans2","native_language"]]


    df_1 = train_data.drop(["ans1","ans2"],axis = 1)
    df_2 = train_data.drop(["sent","ans2"],axis = 1)
    df_3 = train_data.drop(["ans1","sent"],axis = 1)
    df_1.rename(columns = {'sent':'text'}, inplace = True)
    df_2.rename(columns = {'ans1':'text'}, inplace = True)
    df_3.rename(columns = {'ans2':'text'}, inplace = True)


    df_4 = test_data.drop(["ans1","ans2"],axis = 1)
    df_5 = test_data.drop(["sent","ans2"],axis = 1)
    df_6 = test_data.drop(["ans1","sent"],axis = 1)
    df_4.rename(columns = {'sent':'text'}, inplace = True)
    df_5.rename(columns = {'ans1':'text'}, inplace = True)
    df_6.rename(columns = {'ans2':'text'}, inplace = True)


    frames =[df_2, df_3]
    df_train = pd.concat(frames,join = "inner")



    frames =[df_5, df_6]
    df_test = pd.concat(frames,join = "inner")

    df_train.reset_index(inplace=True)

    df_test.reset_index(inplace=True)

    return df_train, df_test

def perform_basic_preprocessing(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame):
    df_train = train_dataframe
    df_test = test_dataframe
    df_train.dropna(axis =0,inplace=True)
    df_train.reset_index(inplace=True)


    df_test.dropna(axis =0,inplace=True)
    df_test.reset_index(inplace=True)

    return df_train, df_test


def load_data(train_path: Optional[str] = None,
              test_path: Optional[str] = None):
    """Load train and test DataFrames. If None, uses config placeholders."""
    train_path = train_path or TRAIN_FILE
    test_path = test_path or TEST_FILE
    train_df = load_csv(train_path)
    test_df = load_csv(test_path)
    train_df, test_df = subset_data(train_df, test_df)
    train_df, test_df = perform_basic_preprocessing(train_df, test_df)
    return train_df, test_df





