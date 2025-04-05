import math
import numpy as np;
from sklearn.preprocessing import StandardScaler

def summarize_dataset(data):
    data = [row[:-1] for row in data]
    summaries = []
    for col in zip(*data):
        mean = sum(col) / len(col)
        stddev = math.sqrt(sum((x - mean) ** 2 for x in col) / (len(col) - 1))
        summaries.append((mean, stddev))
    return summaries

def gaussian_prob(x, mean, stdev):
    if stdev == 0:
        return 1 if x == mean else 0
    exponent = math.exp(- ((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    
def calculate_class_probs(summaries_by_class, row):
   probs = {}
   for class_value, summaries in summaries_by_class.items():
       probs[class_value] = 1
       for i in range(len(summaries)):
           mean, stdev = summaries[i]
           x = row[i]
           probs[class_value] *= gaussian_prob(x, mean, stdev)
   return probs

def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row)
    return separated

def split_data(data, train_ratio=0.7):
    split_index = int(len(data) * train_ratio)
    return data[:split_index], data[split_index:]

def preprocess_data(df):
    df['Gender'] = df['Gender'].apply(lambda x: 0 if x == 'F' else 1)
    df.fillna(df.mean(), inplace=True)
    numeric_columns = ['Age', 'BMI', 'Chol', 'TG', 'HDL', 'LDL', 'Cr', 'BUN']
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    for col in numeric_columns:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df[col] = np.where(z_scores > 3, df[col].mean(), df[col])

    return df

def evaluate(predictions, test_set):
    correct = 0
    for i in range(len(test_set)):
        if predictions[i] == test_set[i][-1]:
            correct += 1
    return correct / len(test_set) * 100
