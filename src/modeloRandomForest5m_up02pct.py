import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from joblib import dump

# Creacion del modelo secundario RandoForest y los pesos de las features

df = pd.read_csv('../data/history/fullhistorywithindicators5m_up02pct.csv')
#df.drop(['', ''], axis=1, inplace=True)
print("INFO [COLUMNS]: Columnas del dataframe del modelo:", df.columns)
X = df.drop(['Price_up_02_pct'], axis=1) 
y = df['Price_up_02_pct']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Creating the model:")
model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=3)
model.fit(X_train, y_train)

# Save the model
#dump(model, './models/random_forest_model5m_PriceUp02pct.joblib')
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Pesos
feature_importances = model.feature_importances_
indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(12, 6))
plt.title("Peso de las caracteristicas")
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), X.columns[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()
