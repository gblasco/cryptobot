import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Creacion del modelo de red neuronal.

df = pd.read_csv('../data/history/fullhistorywithindicators5m_up02pct.csv')
print("INFO [COLUMNS]: Columnas del dataframe del modelo:", df.columns)

# Define target
X = df.drop(['Price_up_02_pct'], axis=1) 
y = df['Price_up_02_pct']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
features = [
            'Quote Asset Volume', 'Number of Trades',
            'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume',
            'Price_to_EMA_20_Ratio', 'Price_to_EMA_50_Ratio',
            'MACD_line', 'MACD_signal', 'MACD_diff',
            'RSI', 'RSI_overbought', 'RSI_oversold', 'RSI_change',
            'price_to_bbh_ratio', 'price_to_bbm_ratio', 'price_to_bbl_ratio', 'ADX', 'Volume_MA10',
            'Log_Volume', 'dist_to_bbh_pct', 'dist_to_bbl_pct',
            'dist_to_support_pct', 'dist_to_resistance_pct',
            'fng_number', 'Dist_to_EMA_20_pct', 'Dist_to_EMA_50_pct',
            'EMA_5min_diff_pct', 'EMA_30min_diff_pct', 'EMA_1h_diff_pct', 'EMA_6h_diff_pct', 'EMA_24h_diff_pct',
            'Close_5min_pct_change', 'Close_30min_pct_change', 'Close_1h_pct_change', 'Close_6h_pct_change', 'Close_24h_pct_change'
        ]

# scalado
scaler = StandardScaler()
X_train.loc[:, features] = scaler.fit_transform(X_train[features])
X_test.loc[:, features] = scaler.transform(X_test[features])

# !!!!!!!!!! lo comento para no machacarlo sin querer !!!!!!!!!!!!!!!!!!!!!!!!!!!!
#joblib.dump(scaler, './models/modeloNeural5m_up02pct_scaled.pkl')
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu',),
    Dense(1, activation='sigmoid')  # Sigmoid activation for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# !!! lo comento para no machacarlo. Para volver a guardarlo tendre que quitar el comentario !!!
#model.save('./models/modeloNeural5m_up02pct_scaled.keras') 
y_pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precision del entrenamieto')
plt.plot(history.history['val_accuracy'], label='Precision de la validacion')
plt.title('Precision del modelo')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Modelo Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
