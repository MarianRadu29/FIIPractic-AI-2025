import os
from src.utils import load_dataset

dataset_path = os.path.join("data","diabetes_dataset.csv")
data = load_dataset(dataset_path)

