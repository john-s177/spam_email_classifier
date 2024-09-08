import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

def printResult(name, accuracy, precision, recall, conf_matrix):
    app = tk.Tk()
    app.geometry('500x500')

    labels = [
        ("Name", name), ("Accuracy", accuracy), 
        ("Precision", precision), ("Recall", recall), 
        ("Confusion Matrix", conf_matrix)
    ]

    for text, value in labels:
        tk.Label(app, text=f"{text} :").pack()
        tk.Label(app, text=str(value)).pack()

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    app.mainloop()

filepath = filedialog.askopenfilename()

# Load the dataset
df = pd.read_csv(filepath)

# Preprocessing
df['text'] = df['text'].str.lower()
df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

df['label_num'] = df['label'].apply(lambda x: 0 if x == 'ham' else 1)
df.drop('label', axis=1, inplace=True)

# Convert text data into numerical features using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['text'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, df['label_num'], test_size=0.2, random_state=42)

# Feature Selection
selector = SelectKBest(score_func=chi2, k=1000)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Classifier Functions
def logisticRegression():
    lr_model = LogisticRegression()
    lr_model.fit(X_train_selected, y_train)
    y_pred = lr_model.predict(X_test_selected)
    printResult("Logistic Regression", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

def svm():
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train_selected, y_train)
    y_pred = svm_model.predict(X_test_selected)
    printResult("SVM", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

def kNN():
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train_selected, y_train)
    y_pred = knn_model.predict(X_test_selected)
    printResult("k-NN", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

def gradientBoosting():
    gb_model = GradientBoostingClassifier()
    gb_model.fit(X_train_selected, y_train)
    y_pred = gb_model.predict(X_test_selected)
    printResult("Gradient Boosting", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

def mlp():
    mlp_model = MLPClassifier()
    mlp_model.fit(X_train_selected, y_train)
    y_pred = mlp_model.predict(X_test_selected)
    printResult("MLP", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

def decisionTree():
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train_selected, y_train)
    y_pred = dt_model.predict(X_test_selected)
    printResult("Decision Tree", accuracy_score(y_test, y_pred), 
                precision_score(y_test, y_pred), recall_score(y_test, y_pred), 
                confusion_matrix(y_test, y_pred))

# GUI for selecting classifiers
app = tk.Tk()
app.geometry('200x300')

buttons = [
    ("Logistic Regression", logisticRegression),
    ("SVM", svm),
    ("Decision Tree", decisionTree),
    ("k-NN", kNN),
    ("Gradient Boosting", gradientBoosting),
    ("MLP", mlp)
]

for text, command in buttons:
    Button(app, text=text, width=20, height=2, command=command).pack(pady=10)

app.mainloop()
