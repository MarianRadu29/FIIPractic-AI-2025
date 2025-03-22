from sklearn.utils import shuffle
import pandas as pd

def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return shuffle(data,random_state=42)