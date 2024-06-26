
from engineBTCHistory import GetDataBTC
from binance.client import Client
from binance.enums import *
import pandas as pd
import ta
import datetime
import matplotlib.pyplot as plt
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator, PercentagePriceOscillator
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import requests
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# Se ejecuta desde el fichero runIndicatorsLive.py
# A tener en cuenta fng_number quitarlo o dejarlo porque no hay datos en 2017 o ponerle un valor neutral.

class IndicatorsLive:

    def convert_to_minutes(self, interval):
        interval_mapping = {
            '1m': 1,
            '5m': 5,
            '30m': 30,
            '1h': 60,
            '1d': 1440
        }
    
        if interval in interval_mapping:
            return interval_mapping[interval]
        else:
            raise ValueError("ERROR : [ Intervalo no válido. ] Por favor, usa uno de los siguientes: '1m', '5m', '30m', '1h', '1d'.")


    def dropColumns(self, df):
        df.drop(['Open', 'High', 'Low', 'Volume', 'date', 'EMA_20', 'EMA_50', 'bbm', 'bbl', 'bbh', 'support', 'resistance', 'Close' ], axis=1, inplace=True)
        print(f"LOG: [Indicators Calculated]:  Drop not necessary columns...")   
        return df
    

    def getEMA(self, df):
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        # cruces de las EMAs
        df['EMA_cross_up'] = ((df['EMA_20'].shift(1) < df['EMA_50'].shift(1)) & (df['EMA_20'] > df['EMA_50']))
        df['EMA_cross_down'] = ((df['EMA_20'].shift(1) > df['EMA_50'].shift(1)) & (df['EMA_20'] < df['EMA_50']))
        # cruces del precio con las EMAs
        df['Price_cross_up_EMA_20'] = ((df['Close'].shift(1) < df['EMA_20'].shift(1)) & (df['Close'] > df['EMA_20']))
        df['Price_cross_down_EMA_20'] = ((df['Close'].shift(1) > df['EMA_20'].shift(1)) & (df['Close'] < df['EMA_20']))
        df['Price_cross_up_EMA_50'] = ((df['Close'].shift(1) < df['EMA_50'].shift(1)) & (df['Close'] > df['EMA_50']))
        df['Price_cross_down_EMA_50'] = ((df['Close'].shift(1) > df['EMA_50'].shift(1)) & (df['Close'] < df['EMA_50']))
        df['Price_higher_EMA_20'] = df['Close'] > df['EMA_20']
        df['Price_higher_EMA_50'] = df['Close'] > df['EMA_50']
        df['Price_lower_EMA_20'] = df['Close'] < df['EMA_20']
        df['Price_lower_EMA_50'] = df['Close'] < df['EMA_50']
        df['Price_to_EMA_20_Ratio'] = df['Close'] / df['EMA_20']
        df['Price_to_EMA_50_Ratio'] = df['Close'] / df['EMA_50']
        df['Dist_to_EMA_20_pct'] = (df['Close'] - df['EMA_20']) / df['EMA_20'] * 100
        df['Dist_to_EMA_50_pct'] = (df['Close'] - df['EMA_50']) / df['EMA_50'] * 100
        df['EMA_5min_diff_pct'] = (df['EMA_20'].diff(1) / df['EMA_20']) * 100
        df['EMA_30min_diff_pct'] = (df['EMA_20'].diff(6) / df['EMA_20']) * 100
        df['EMA_1h_diff_pct'] = (df['EMA_20'].diff(12) / df['EMA_20']) * 100
        df['EMA_6h_diff_pct'] = (df['EMA_20'].diff(72) / df['EMA_20']) * 100
        df['EMA_24h_diff_pct'] = (df['EMA_20'].diff(288) / df['EMA_20']) * 100
        ema_diff_features = ['EMA_5min_diff_pct', 'EMA_30min_diff_pct', 'EMA_1h_diff_pct', 'EMA_6h_diff_pct', 'EMA_24h_diff_pct']
        #df[ema_diff_features] = df[ema_diff_features].fillna(0)
        # debo aniador tb EMA50 diff pct??? segun el peso
        return df

    
    def addScaler(self, df):
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
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        return df
        
    
    def addNums(self, df):
        bool_columns = [
            'EMA_cross_up', 'EMA_cross_down',
            'Price_cross_up_EMA_20', 'Price_cross_down_EMA_20',
            'Price_cross_up_EMA_50', 'Price_cross_down_EMA_50',
            'Price_higher_EMA_20', 'Price_higher_EMA_50',
            'Price_lower_EMA_20', 'Price_lower_EMA_50',
            'bb_cross_upper', 'bb_cross_lower'
            ]
        # Convertir cada columna booleana a entero (1 o 0)
        for column in bool_columns:
            df[column] = df[column].astype(int)
        
        return df
    
    
    def addRSI(self, df):
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_overbought'] = (df['RSI'] > 70).astype(int)  # Sobrecompra
        df['RSI_oversold'] = (df['RSI'] < 30).astype(int)   # Sobreventa
        df['RSI_change'] = df['RSI'].diff()  # diferencia con respecto al anterior en el RSI
        return df
    
    def addMACD(self, df):
        df['MACD_line'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd()
        df['MACD_signal'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_signal()
        df['MACD_diff'] = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9).macd_diff()
        return df
    
    def addBollinger(self, df):
        indicator_bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['bbm'] = indicator_bb.bollinger_mavg()   # Banda media
        df['bbh'] = indicator_bb.bollinger_hband()  # Banda superior
        df['bbl'] = indicator_bb.bollinger_lband()  # Banda inferior
        # Cruces
        df['bb_cross_upper'] = (df['Close'] > df['bbh'])
        df['bb_cross_lower'] = (df['Close'] < df['bbl'])
        # distancia porcentual
        df['dist_to_bbh_pct'] = (df['Close'] - df['bbh']) / df['bbh'] * 100
        df['dist_to_bbl_pct'] = (df['Close'] - df['bbl']) / df['bbl'] * 100
        df['price_to_bbh_ratio'] = df['Close'] / df['bbh']
        df['price_to_bbl_ratio'] = df['Close'] / df['bbl']
        df['price_to_bbm_ratio'] = df['Close'] / df['bbm'] 
        return df

    def addADX(self, df):
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
        return df
    
    def addVolume(self, df):
        df['Volume_MA10'] = df['Volume'].rolling(window=10).mean()
        df['Log_Volume'] = np.log(df['Volume'] + 1)  # Sumamos 1 para los casos donde volumen sea 0
        return df

    def addCloseDeltas(self, df):
        df['Close_5min_pct_change'] = df['Close'].pct_change(1) * 100  
        df['Close_30min_pct_change'] = df['Close'].pct_change(6) * 100  
        df['Close_1h_pct_change'] = df['Close'].pct_change(12) * 100  
        df['Close_6h_pct_change'] = df['Close'].pct_change(72) * 100   
        df['Close_24h_pct_change'] = df['Close'].pct_change(288) * 100
        return df

    def calculate_support_resistance(self, df, window):
        #El valor window = 20 me ha funcionado bien.
        # minimos y maximos rodantes
        min_roll = df['Close'].rolling(window=window, center=True).min()
        max_roll = df['Close'].rolling(window=window, center=True).max()
        # Identificar los minimos locales como soporte y maximos como resistencia
        df['support'] = df['Close'][df['Close'] == min_roll]
        df['resistance'] = df['Close'][df['Close'] == max_roll]
        # Relleno nulos con el valor anterior
        df['support'].fillna(method='ffill', inplace=True)
        df['resistance'].fillna(method='ffill', inplace=True)
        # Calcular distancias a soporte y resistencia en %
        df['dist_to_support_pct'] = (df['Close'] - df['support']) / df['Close'] * 100
        df['dist_to_resistance_pct'] = (df['resistance'] - df['Close']) / df['Close'] * 100
        # Ver si hay cruce
        df['cross_support'] = ((df['Close'].shift(1) > df['support'].shift(1)) & (df['Close'] < df['support'])).astype(int)
        df['cross_resistance'] = ((df['Close'].shift(1) < df['resistance'].shift(1)) & (df['Close'] > df['resistance'])).astype(int)
        return df

    def index_greed_fear(self, df):
        # No tengo valores para 2017... de momento he dejado NaN pero igual hay que cambiarlo por neutral o tratarlo de otra forma
        # probablemente deberia cargarme esos datos historicos donde no tengo valor
        response = requests.get("https://api.alternative.me/fng/?limit=0&format=json") 
        if response.status_code == 200:
            data = response.json()
            data_list = data.get('data', [])
            df2 = pd.DataFrame(data_list)
            df2['date'] = pd.to_datetime(df2['timestamp'], unit='s')
            df2['date'] = df2['date'].dt.tz_localize('UTC').dt.tz_convert('Europe/London')
            df2.drop(columns=['timestamp','time_until_update', 'value_classification'], inplace=True)
            df2['date'] = df2['date'].dt.date
            df['date'] = pd.to_datetime(df['Time'])
            df['date'] = df['date'].dt.date
            merged_df = pd.merge(df, df2, on='date', how='left')
            #merged_df = merged_df.rename(columns={'value_classification': 'fng_index'})
            merged_df = merged_df.rename(columns={'value': 'fng_number'})
            print(f"LOG: [Indicators Calculated]: Indicador de miedo y codicia calculado")   
            return merged_df

        else:
            print("[LOG]: Error en la solicitud de indice de fear and greed:", response.status_code)
            return df

    
    def addTarget(self, df, look_ahead_intervals, pct):
        # look_ahead_intervals = 1 # funciona bien y 3 mejor
        # look_ahead_intervals = 3  # 6 periodos de 5 minutos cada uno hacen 30 minutos
        # cambio porcentual entre el precio actual y el precio look_ahead_intervals
        # pct 0.2 y 0.5 funciona bien
        df['Price_change_pct'] = ((df['Close'].shift(-look_ahead_intervals) - df['Close']) / df['Close']) * 100
        # si el cambio porcentual es mayor que parametro que recibo de terminal (arg)
        df['Price_up_02_pct'] = df['Price_change_pct'] > pct
        df['Price_up_02_pct'] = df['Price_up_02_pct'].astype(int)
        return df

    def balance_data(self, df):
        # Cuenta la cantidad de casos en cada clase
        count_class_0, count_class_1 = df['Price_up_02_pct'].value_counts()
        # Divido el dataframe por clase
        df_class_0 = df[df['Price_up_02_pct'] == 0]
        df_class_1 = df[df['Price_up_02_pct'] == 1]
        # submuestreo aleatorio en la clase mayoritaria
        df_class_0_under = df_class_0.sample(count_class_1)
        df_balanced = pd.concat([df_class_0_under, df_class_1], axis=0)
        print('Random under-sampling:')
        print(df_balanced['Price_up_02_pct'].value_counts())
        return df_balanced

    def plot_support_resistance(self, df):
        plt.figure(figsize=(10, 7))
        df = df.tail(400)
        plt.plot(df.index, df['Close'], label='Close', color='black')
        plt.scatter(df.index, df['support'], label='Support', color='green', marker='o')
        plt.scatter(df.index, df['resistance'], label='Resistance', color='red', marker='o')
        # Marcar los cruces de soporte y resistencia
        plt.scatter(df.index[df['cross_support'] == 1], df['Close'][df['cross_support'] == 1], label='Cross Support', color='blue', marker='^')
        plt.scatter(df.index[df['cross_resistance'] == 1], df['Close'][df['cross_resistance'] == 1], label='Cross Resistance', color='orange', marker='^')
        plt.title('Support and Resistance Levels')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def dropNulls(self, df):
        # fng number tiene nulos porque no tenemos datos hasta 2018, asi que ya sabemos esos nulos hasta 2018.
        # podriamos ponerle un neutral para los datos anteriores pero por ahora he decidido dejarlo asi.
        rowsBefore = df.shape
        df = df.dropna()
        rowsAfter = df.shape
        print(f"LOG: [Quitar nulos del dataset]: El dataframe tenia ({rowsBefore}) y al quitar los nulos: ({rowsAfter})")
        return df
    
    def addModelScaled(self, df):
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
        
        return df

    def getIndicatorsCalculated(self, interval, windowrs, look_ahead_intervals, pct):
        df = pd.read_csv(f'../data/live/live{interval}.csv')
        df = self.getEMA(df)
        #df = self.getSMA(df)
        df = self.addRSI(df)
        df = self.addMACD(df)
        df = self.addBollinger(df)
        df = self.addADX(df)
        df = self.addVolume(df)
        df = self.addCloseDeltas(df)
        df = self.calculate_support_resistance(df, windowrs)
        df = self.index_greed_fear(df)
        #df = self.addTarget(df, look_ahead_intervals, pct) necesario para entrenar modelo en el historico
        xxxdfval = df
        # solo para validar
        xxxdfval = self.dropNulls(xxxdfval)
        xxxdfval.to_csv(os.path.join('.', 'xxxvalidate.csv'), index=True) # se puede comentar es solo para validar datos pasados pegandole la prediccion
        # aqui acaba la validacion
        df = self.dropColumns(df)
        df = self.dropNulls(df)
        #df = self.balance_data(df)
        #df = self.addScaler(df)
        df = self.addNums(df)
        #df = self.addModelScaled(df)
        #df = self.plot_support_resistance(df)
        df.set_index('Time', inplace=True) # ver si lo elimino de mis datos
        # Up or down
        #df.to_csv(os.path.join('../data/history', 'fullhistorywithindicators' + interval + '.csv'), index=False)
        #print(f"LOG: [Indicators Calculated]:  ../data/history/fullhistorywithindicators{interval}.csv")
        # up 0.2%
        df.to_csv(os.path.join('../data/live', 'livewithindicators' + interval + '_up02pct.csv'), index=False)
        print(f"LOG: [Indicators Calculated]:  ../data/live/livewithindicators{interval}_up02pct.csv")
        return df
    
