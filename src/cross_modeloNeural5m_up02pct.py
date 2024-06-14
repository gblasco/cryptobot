import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Crossvalidation de mi modelo de redNeuronal de 5 minutos
# Predice si va a subir un 0.5% (aunque el nombre del campo esta dice 02 esta desfasado habria que cambiarlo)
# Ejecucion: python cross_modeloNeural5m_up02pct.py

df = pd.read_csv('../data/history/fullhistorywithindicators5m_up02pct.csv')
print("INFO [COLUMNS]: Columnas del dataframe del modelo:", df.columns)

# Target
X = df.drop(['Price_Up_0_2_Percent'], axis=1)
y = df['Price_Up_0_2_Percent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
input_shape = (X_train.shape[1],)

def create_model(input_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Cross validation nerual
pipeline = Pipeline([
    ('model', KerasClassifier(model=create_model, input_shape=input_shape, epochs=10, batch_size=32, verbose=1))
])

# Cross validation
cv_score = cross_val_score(pipeline, X_train, y_train, cv=5)
print("Cross-Validation Mean Accuracy:", np.mean(cv_score))

# Hyperparametros
param_grid = {
    'model__epochs': [10, 20],
    'model__batch_size': [32, 64]
}

grid_searchcv = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_searchcv.fit(X_train, y_train)
print("Mejores parametros:", grid_searchcv.best_params_)
print("Mejor Cross-Validation Score:", grid_searchcv.best_score_)

# Evaluacion del modelo
topmodel = grid_searchcv.best_estimator_
y_pred = topmodel.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test precision:", test_accuracy)
print("Test precision: {:.2f}%".format(test_accuracy * 100))

# topmodel.named_steps['model'].model.save('topmodel_neural_network_model.h5')

# Entrenar el mejor modelo
topmodel.fit(X_train, y_train)
train_size, train_scores, test_scores = learning_curve(topmodel, X_train, y_train, cv=5)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(9, 7))
plt.plot(train_size, train_mean, label='Puntuacion entrenamiento')
plt.plot(train_size, test_mean, label='Cross-validation puntuacion')
plt.fill_between(train_size, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_size, test_mean - test_std, test_mean + test_std, alpha=0.1)
plt.title('Curva de entrenamiento')
plt.xlabel('Numero de ejemplos de entrenamiento')
plt.ylabel('Puntuacion')
plt.legend()
plt.grid()
plt.show()

y_pred = topmodel.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Clase 0', 'Clase 1'])
plt.figure(figsize=(9, 7))
disp.plot(cmap=plt.cm.Blues) 
plt.title('Matriz de confusion')
plt.show()



