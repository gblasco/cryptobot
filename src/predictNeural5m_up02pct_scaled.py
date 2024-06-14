import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Programa para realizar la prediccion, necesita que se ejecute antes el calculo de indicadores

df = pd.read_csv('../data/live/livewithindicators5m_up02pct.csv')
# Cargar el modelo
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

try:
    # Cargar el scaler para predecir
    scaler = joblib.load('./models/modeloNeural5m_up02pct_scaled.pkl')
    df.loc[:, features] = scaler.transform(df[features])
except Exception as e:
    print(f"Error al cargar el scaler o al transformar los datos: {e}")
        
model = load_model('./models/modeloNeural5m_up02pct_scaled.keras') 
# Hacer la prediccion
prediction = model.predict(df.tail(1)) # cojo el ultimo registro
print("La prediccion es:", prediction)

# solo para validar datos
# dfval = pd.read_csv('xxxvalidate.csv')
# dfval['predict'] = prediction
# dfval.to_csv('xxxvalidatepredicts.csv',  index=False)