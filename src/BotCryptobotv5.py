import csv
import sys
from time import sleep
import engineBTCLive
from engineIndicatorsLive import IndicatorsLive
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
import time
import datetime
from binance.enums import *

# Parametros necesarios: <interval> <live?> [limit] -> 5m True 289
# Ejecuta en orden primero: 1- engineBTCLive.py y tengo que traer minimo 289 registros para que pueda calcular todos los datos
# pasados que necesita el dato live para no tener nulos.
# 2 - Ejecuta engineIndicatorsLive.py para calcular todos los indicadores necesarios de mi modelo.
# 3 - Ejecuta predictNeural5m_up02pct_scaled.py, hace el escalado de mis datos y la prediccion
# 4 - Compra o no en base a mi prediccion

# Variable global para comprar cuando se pone a True
buysignal = False

def wait_until_next_predict(secs):
    now = datetime.datetime.now()
    # cuanto falta para el proximo intervalo de 5 minutos para alinearme con la hora
    waitsecs = secs - (now.minute * 60 + now.second) % secs
    print(f"LOG: [ Esperando {waitsecs} segundos]")
    time.sleep(waitsecs)  # Espera hasta el siguente intervalo de 5 minutos
    return waitsecs

def predictBuy(df, interval, windowrs, lookahead, pct, predictbuypct):
    # cambiarlo para usar el predict que tengo en otro fichero
    print(df[['Time', 'Close']].tail(3))
    dfwithtime = df[['Time', 'Close']].tail(1)
    # runIndicatorsLive / engineIndicatorsLive
    bot = IndicatorsLive()
    df = bot.getIndicatorsCalculated(interval, windowrs, lookahead, pct)
    #df = pd.read_csv('../data/live/livewithindicators5m_up02pct.csv')
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

    prediction = model.predict(df.tail(1)) # cojo el ultimo registro 
    #print("La prediccion es:", prediction)
    # aniado una columna prediction al df que ya tenia la hora y el precio.
    dfwithtime['Prediction'] = prediction[0]
    lasttime = dfwithtime['Time'].iloc[0]
    lastclose = dfwithtime['Close'].iloc[0]
    lastpred = dfwithtime['Prediction'].iloc[0]
    print(dfwithtime)
    #print(f"Los valores son Time: {lastTime} y Close: {lastClose}")
    # predictbuypct is the pct to buy. ie. 0.75
    if lastpred > predictbuypct:
        #dfwithtime['Action'] = 'B'
        print(f"Compro porque la prediccion es alta: {lastpred}")
        log_append_to_csv(dfwithtime, 'log_all_orders.csv')
        return True, lasttime, lastclose, lastpred
    else:
        #dfwithtime['Action'] = 'N'
        print(f"No comprar porque la prediccion es baja: {lastpred}")
        log_append_to_csv(dfwithtime, 'log_all_orders.csv')
        return False, lasttime, lastclose, lastpred

def log_append_to_csv(df, file):
    df.to_csv(file, mode='a', header=False, index=False)


def buy_at_market(client, symbol, quantity, minsorder, lastClose):
    try:
        # Comprar a mercado
        buy_order = client.order_market_buy(symbol=symbol, quantity=quantity)
        order_id = buy_order['orderId']
        print(f"Orden de compra puesta, ID: {order_id}")
        estimatedprice = quantity * lastClose
        print(f"Coste estimado en USDT: {estimatedprice}")
        start_time = time.time()

        while True:
            order_status = client.get_order(symbol=symbol, orderId=order_id)
            if order_status['status'] == 'FILLED':
                print("La orden ha sido completada.")
                break
            elif time.time() - start_time > minsorder * 60:
                print("20 minutos han pasado, cancelando la orden que no se ha llenado y saliendo...")
                client.cancel_order(symbol=symbol, orderId=order_id)
                return None
            else:
                print("La orden aun no esta completada, esperando...")
                time.sleep(1)  # Espera un segundo antes de comprobar de nuevo

        # si esto funciona calcular el precio de compra ponderado 
        print(f"El precio de compra de la orden ha sido: {buy_order['price']}")
        if 'fills' in buy_order and len(buy_order['fills']) > 0:
            buy_price = float(buy_order['fills'][0]['price'])
        else:
            buy_price = buy_order['price']  # Usa lastClose como valor predeterminado si no hay fills disponibles
        
        print(buy_order)
        print(order_status)
        print(f"Buy price is: {buy_price} y lastclose era {lastClose}")
        sell_quantity = float(order_status['executedQty'])
        take_profit_price = "{:.2f}".format(buy_price * 1.004) # cambiar a 1.005 usar el parametro
        stop_loss_price = "{:.2f}".format(buy_price * 0.994) # cambiar a 0.995 usar el parametro
        stop_limit_price = "{:.2f}".format(buy_price * 0.9941) # lo mismo
        
        # ordenes de take profit y stop loss
        orderprofitsell = client.order_limit_sell(symbol=symbol, quantity=sell_quantity, price=take_profit_price)
        orderlosssell = client.create_order(
            symbol=symbol,
            side='SELL',
            type='STOP_LOSS_LIMIT',
            timeInForce='GTC',
            quantity=sell_quantity,
            price=stop_loss_price, # precio venta
            stopPrice=stop_limit_price # activacion
        )
        #open_orders = client.get_open_orders()
        #print(open_orders)
        print(f"Tomar beneficio orden ID: {orderprofitsell['orderId']}")
        print(f"Stop loss orden ID: {orderlosssell['orderId']}")
        # Reiniciar el contador
        start_time = time.time()  
        print(f"Esperando a que alcance los precios o se vendera tras: {minsorder} minutos")
        # Verificar estado de las nuevas ordenes puestas
        while True:
            if time.time() - start_time > minsorder * 60:
                print("20 minutos han pasado desde ordenes de take profit y stop loss, cancelando ambas ordenes...")
                ordersell = client.order_market_sell(symbol=symbol, quantity=quantity)
                client.cancel_order(symbol=symbol, orderId=orderprofitsell['orderId'])
                client.cancel_order(symbol=symbol, orderId=orderlosssell['orderId'])
                order_idsell = ordersell['orderId']
                print(f"Orden de venta a mercado creada: {order_idsell}")
                while True:
                    ordersell_status = client.get_order(symbol=symbol, orderId=order_idsell)
                    if ordersell_status['status'] == 'FILLED':
                        if 'fills' in ordersell and len(ordersell['fills']) > 0:
                            sell_price = ordersell['fills'][0]['price']
                        else:
                            sell_price = '0'
                        try:
                            sell_price = float(sell_price)
                        except ValueError:
                            sell_price = 0.0  # Controlar esta excepcion poniendo un default

                        print(f"La orden de venta a mercado completada al precio: {sell_price}")
                        print(ordersell_status)
                        print(ordersell)
                        #meterlo funcion
                        current_time = datetime.datetime.now()
                        balance = buy_price - sell_price
                        data = {
                        'time': current_time,
                        'buy_price': buy_price,
                        'sell_price': sell_price,
                        'balance' : balance
                        }
                        dfbalance = pd.DataFrame([data])
                        log_append_to_csv(dfbalance, 'log_buysell_orders.csv')
                        print("Guardando la operacion de compra venta")
                        # meterlo funcion
                        return None
                    else:
                        print("Esperando completar orden de venta a mercado")
                        sleep(5)
            
            take_profit_status = client.get_order(symbol=symbol, orderId=orderprofitsell['orderId'])['status']
            stop_loss_status = client.get_order(symbol=symbol, orderId=orderlosssell['orderId'])['status']

            if take_profit_status == 'FILLED':
                print("Orden de take profit completada.")
                client.cancel_order(symbol=symbol, orderId=orderlosssell['orderId']) # cancelo el stop loss sell
                #meterlo funcion
                # current_time = datetime.datetime.now()
                # balance = buy_price - sell_price
                # data = {
                # 'time': current_time,
                # 'buy_price': buy_price,
                # 'sell_price': sell_price,
                # 'balance' : balance
                # }
                # dfbalance = pd.DataFrame([data])
                # log_append_to_csv(dfbalance, 'log_buysell_orders.csv')
                # print("Guardando la operacion de compra venta")
                # meterlo funcion
                return None
            if stop_loss_status == 'FILLED':
                print("Orden de stop loss completada.")
                client.cancel_order(symbol=symbol, orderId=orderprofitsell['orderId']) # cancelo el profitsell
                                #meterlo funcion
                # current_time = datetime.datetime.now()
                # balance = buy_price - sell_price
                # data = {
                # 'time': current_time,
                # 'buy_price': buy_price,
                # 'sell_price': sell_price,
                # 'balance' : balance
                # }
                # dfbalance = pd.DataFrame([data])
                # log_append_to_csv(dfbalance, 'log_buysell_orders.csv')
                # print("Guardando la operacion de compra venta")
                # meterlo funcion
                return None

            #print("Ninguna de las ordenes ha sido completada aun, revisando de nuevo...")
            time.sleep(5)  # Espera 5 segundos antes de verificar de nuevo
       
    except BinanceAPIException as e:
        print(f"Error en la API: {e.message}")
    except Exception as e:
        print(f"Error: {e}")

    current_time = datetime.datetime.now()
    balance = buy_price - sell_price
    data = {
    'time': current_time,
    'buy_price': buy_price,
    'sell_price': sell_price,
    'balance' : balance
    }
    
    dfbalance = pd.DataFrame([data])
    log_append_to_csv(dfbalance, 'log_buysell_orders.csv')
    print("Logeando la operacion de compra venta")
    
    
def sell_at_market(client, symbol, quantity):
    order = client.order_market_sell(symbol=symbol, quantity=quantity)
    return order

def cancel_order(client, symbol, orderid):
    order = client.cancel_order(symbol=symbol, orderId=orderid)

def cancel_all_orders(client):
    try:
        # ordenes abiertas
        open_orders = client.get_open_orders()
        print(f'Numero de ordenes abiertas: {len(open_orders)}')
        for order in open_orders:
            print(order)
        
        # Cancelar ordenes abiertas
        for order in open_orders:
            result = client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
            print(f'Orden cancelada: {result}')
    except BinanceAPIException as e:
        print(f"API Error: {e.status_code} - {e.message}")
    except BinanceRequestException as e:
        print(f"Request Error: {e.status_code} - {e.message}")
    except Exception as e:
        print(f"General Error: {str(e)}")    
    
def cancel_buy_all_orders(client):
    try:
        open_orders = client.get_open_orders()
        print(f'Numero de ordenes abiertas totales: {len(open_orders)}')
        buy_orders_count = sum(1 for order in open_orders if order['side'] == 'BUY')
        print(f"Numero de ordenes de compra activas: {buy_orders_count}")
        for order in open_orders:
            print(order)
    except BinanceAPIException as e:
        print(f"API Error: {e}")
        open_orders = [] 
        
    for order in open_orders:
        if order['side'] == 'BUY':
            try:
                result = client.cancel_order(symbol=order['symbol'], orderId=order['orderId'])
                print(f"Orden de compra cancelada: {result}")
            except BinanceAPIException as e:
                print(f"Error al cancelar la orden {order['orderId']}: {e}")

def buy_order(client, lastTime, lastClose, prediction, quantity, pct, minsorder):
    # esto tengo que hacerlo mas parametrizable, para decir cuanto comprar y %s de ganancia y perdida quiero.
    # ahora mismo busco 0.4% de subida y compro 0.1 btc dandole 20mins de tiempo
    global buysignal
    symbol = "BTCUSDT"
    if buysignal:
        cancel_buy_all_orders(client)
        cancel_all_orders(client)
        buy_at_market(client, symbol, quantity, minsorder, lastClose)
        open_orders = client.get_open_orders()
        print(open_orders)
        print(f'Numero de ordenes activas: {len(open_orders)}')              
        account_balance = client.get_asset_balance(asset='BTC')
        btcbalance = account_balance['free']
        account_balance = client.get_asset_balance(asset='USDT')
        usdtbalance = account_balance['free']
        print(f"BTC Balance: {btcbalance} available")
        print(f"USDT Balance: {usdtbalance} available")
        cancel_all_orders(client)
        print("Cancelando todas las ordenes por si se me ha quedado alguna abierta")
        # Balance inicial
        initial_btc_balance = 0.01212585
        initial_usdt_balance = 19.33326752
        usdt_balance = usdtbalance
        btc_balance = btcbalance
        # guardo en csv un resumen de la operacion
        #log_order_to_csv(lastTime, lastClose, prediction, order_id, compra_price, target_price, stop_price, final_status, usdt_balance)
        buysignal = False

def main():
    global buysignal
    interval = sys.argv[1] # '5m'
    live = sys.argv[2].lower() == 'true'
    limit = 289
    if len(sys.argv) > 3:
        limit = int(sys.argv[3])
    # engineBTCLive
    if len(sys.argv) < 3:
        print("ERROR: [ Parametros necesarios: <interval> <live?> [limit] ]: Por ejemplo: " + sys.argv[0] + " 5m " + "True " + "500 (limit solo se usa para limitar datos live) ")
        sys.exit(1)  # Termina el programa con un error
    secs = 300 # importante ! poner 300 si quiero cuadrar la hora con los intervalos. Poner 6o si quiero que ejecute cada min   
    interval = '5m'
    windowrs = 20
    lookahead = 4
    live = True
    pct = 0.5 #entrenamiento modelo
    btcamount = 0.001
    pctup = 0.4 # mi objetivo por ahora
    minsorder = 25 # cuantos minutos de orden normalmente 20
    predictbuypct = 0.78 #  0.7 for test, then 0.75 or 0.78
    # wallet
    load_dotenv()  # This loads the environment variables from a .env file

    #api_key = os.getenv('BINANCE_API_KEY_TEST')
    #api_secret = os.getenv('BINANCE_API_SECRET_TEST')
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = Client(api_key, api_secret)
    #client.API_URL = 'https://testnet.binance.vision/api'
    
    while True:
        wait_until_next_predict(secs)
        btc = engineBTCLive.GetDataBTC()
        df = btc.getBTCData(interval, live, limit)
        buysignal, lasttime, lastclose, lastpred = predictBuy(df,interval, windowrs, lookahead, pct, predictbuypct)
        if buysignal:
            buy_order(client, lasttime, lastclose, lastpred, btcamount,pctup, minsorder)
        
if __name__ == "__main__":
    main()


