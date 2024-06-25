import time
from binance.client import Client
import os
from dotenv import load_dotenv
from binance.enums import *
import pandas as pd

def wallet():
    
    def log_append_to_csv(df, file):
        df.to_csv(file, mode='w', header=False, index=False)
        
    load_dotenv() 
    #api_key = os.getenv('BINANCE_API_KEY_TEST')
    #api_secret = os.getenv('BINANCE_API_SECRET_TEST')
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    client = Client(api_key, api_secret)
    #client.API_URL = 'https://testnet.binance.vision/api'

    btc_balance_dict = client.get_asset_balance(asset='BTC')
    btc_balance = float(btc_balance_dict['free'])
    print(f"BTC Balance: {btc_balance} available")
    usdt_balance_dict = client.get_asset_balance(asset='USDT')
    usdt_balance = float(usdt_balance_dict['free'])
    print(f"USDT Balance: {usdt_balance} available")
    ticker = client.get_symbol_ticker(symbol="BTCUSDT")
    btc_price = float(ticker['price'])
    btc_usdt = btc_balance*btc_price
    total_balance_usd = round(float(btc_usdt + usdt_balance),2)
    data = {
        'btc_balance': btc_balance,
        'btc_usdt' : btc_usdt,
        'usdt_balance': usdt_balance,
        'usdt_usdt': usdt_balance,
        'btc_price': btc_price,
        'total_balance_usd' : total_balance_usd
    }
    df = pd.DataFrame([data])
    log_append_to_csv(df, '../data/csv/wallet.csv')
    return btc_price
    
def main():
    return wallet()

if __name__ == "__main__":
    main()