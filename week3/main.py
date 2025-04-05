import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils import *
from src.bayes import *
import math


def main():
    
    df = pd.read_csv("data/Diabetes Classification.csv")
    df = shuffle(df, random_state=42)

    # Preprocesarea datelor
    df = preprocess_data(df)

    data = df.drop(columns=["Nr. Crt."]).values.tolist()
    train_data, test_data = split_data(data, 0.75)
    
    # Bayes Naiv
    predictions_naive = bayes_naive(train_data, test_data)
    accuracy_naive = evaluate(predictions_naive, test_data)
    print(f"Acuratetea cu Bayes Naive: {accuracy_naive:.2f}%")
    
    # Matricea de confuzie pentru Bayes Naiv
    y_true_naive = [row[-1] for row in test_data]  # Etichetele reale (ultimul element din fiecare rand de test_data)
    cm_naive = confusion_matrix(y_true_naive, predictions_naive)
    disp_naive = ConfusionMatrixDisplay(confusion_matrix=cm_naive, display_labels=['Class 0', 'Class 1'])
    disp_naive.plot(cmap='Reds')
    plt.title("Matricea de Confuzie - Bayes Naive")
    plt.savefig("confusion_matrix_naive.png")  

    # Bayes Optimal
    predictions_optimal = bayes_optimal(train_data, test_data)
    accuracy_optimal = evaluate(predictions_optimal, test_data)
    print(f"Acuratetea cu Bayes Optimal: {accuracy_optimal:.2f}%")
    
    # Matricea de confuzie pentru Bayes Optimal
    y_true_optimal = [row[-1] for row in test_data]  # Etichetele reale (ultimul element din fiecare rand de test_data)
    cm_optimal = confusion_matrix(y_true_optimal, predictions_optimal)
    disp_optimal = ConfusionMatrixDisplay(confusion_matrix=cm_optimal, display_labels=['Class 0', 'Class 1'])
    disp_optimal.plot(cmap='Blues')
    plt.title("Matricea de Confuzie - Bayes Optimal")
    plt.savefig("confusion_matrix_optimal.png")  


if __name__ == "__main__":
    main()
