import pandas as pd
from sklearn.preprocessing import StandardScaler  #scalarea datelor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from src.knn import knn_predict,plot_misclassified_points,plot_classified_points
import sys

df = pd.read_csv("data/dataset_hipertensiune3.csv")
X = df[["Varsta","IMC", "Colesterol"]].values
y = df["Hipertensiune"].values

k = 5
if len(sys.argv)>1:
    k = int(sys.argv[1])
print(k)
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


y_pred = knn_predict(X_train,y_train,X_test,k)
plot_misclassified_points(X_test,y_train,y_pred)
plot_classified_points(X_train,X_test,y_train,y_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratetea modelului: {accuracy:.4f}")

