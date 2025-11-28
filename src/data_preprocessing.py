import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path):
    data = pd.read_csv(path)
    return data

def split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)
