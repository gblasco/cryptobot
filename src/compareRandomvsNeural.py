import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier

# Son modelos obsoletos, solo calculan si va a subir o bajar el precio. Con un 64% de precision
# Ejecucion: python compareRandomvsNeural.py

# Cargar dataset
df = pd.read_csv('../data/history/fullhistorywithindicators5m.csv')

# target de modelo
X = df.drop(['Price_Up'], axis=1)
y = df['Price_Up']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def create_model():
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# crear pipelines para cada modelo
pipeline_neural = Pipeline([('model', KerasClassifier(model=create_model, epochs=30, batch_size=64, verbose=1))])
pipeline_randomforest = Pipeline([('model', RandomForestClassifier(n_estimators=100, random_state=42, verbose=3))])
models = {'Neural Network': pipeline_neural, 'Random Forest': pipeline_randomforest}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    results[name] = {'Precision': accuracy, 'Clasificacion reporte': classification_rep}

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
for name, model in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Ratio falsos positivos')
plt.ylabel('Verdaderos positivos')
plt.title('ROC')
plt.legend(loc="lower right")

plt.subplot(2, 2, 2)
for name, model in models.items():
    precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(recall, precision, label=name)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()

plt.subplot(2, 1, 2)
for name, model in models.items():
    plt.hist(model.predict_proba(X_test)[:, 1], bins=25, alpha=0.5, label=name, density=True)
plt.xlabel('Probabilidad predict')
plt.ylabel('Densidad')
plt.title('Distribucion de las predicciones')
plt.legend()

plt.tight_layout()
plt.show()

for name, result in results.items():
    print(f'{name} Resultados:')
    for key, value in result.items():
        print(f'{key}: {value}')
    print('\n')

# Se comparan precisiones del modelo
plt.figure(figsize=(8, 6))
model_names = list(results.keys())
model_accuracies = [result['Accuracy'] for result in results.values()]
plt.bar(model_names, model_accuracies, color=['blue', 'green'])
plt.xlabel('Modelo')
plt.ylabel('Precision')
plt.title('Precision del modelo')
plt.ylim(0, 1)
plt.show()
