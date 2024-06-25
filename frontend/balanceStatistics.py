import csv
import pandas as pd
from datetime import datetime, timedelta

def balanceStatistics():
    
    def log_write_to_csv(file, headers, data):
        df = pd.DataFrame([data], columns=headers)
        df.to_csv(file, index=False)
        
    df = pd.read_csv('../src/log_buysell_orders.csv', header=None, names=['date', 'buy_price', 'sell_price', 'difference'])

    df['date'] = pd.to_datetime(df['date'])
    total_sum = round(df['difference'].sum(),2)
    total_trades = len(df)
    total_sum_usdt = total_sum * 0.001
    # Operaciones ganadoras
    win_operations = len(df[df['difference'] > 0])
    pct_win = (win_operations / total_trades) * 100 if total_trades > 0 else 0

    # anio
    # year_back = datetime.now() - timedelta(days=365)
    # yeardf = df[df['date'] >= year_back]
    # sum_year = yeardf['difference'].sum()
    # total_trades_year = len(yeardf)
    # win_trades_year = len(yeardf[yeardf['difference'] > 0])
    # pct_win_year = (win_trades_year / total_trades_year) * 100 if total_trades_year > 0 else 0
    # sum_year_usdt = sum_year * 0.001
    # mes
    month_back = datetime.now() - timedelta(days=30)
    monthdf = df[df['date'] >= month_back]
    sum_month = round(monthdf['difference'].sum(),2)
    total_trades_month = len(monthdf)
    win_trades_month = len(monthdf[monthdf['difference'] > 0])
    pct_win_month = (win_trades_month / total_trades_month) * 100 if total_trades_month > 0 else 0
    sum_month_usdt = sum_month * 0.001
    # Semana
    week_back = datetime.now() - timedelta(days=7)
    weekdf = df[df['date'] >= week_back]
    sum_week = round(weekdf['difference'].sum(),2)
    total_trades_week = len(weekdf)
    win_trades_week = len(weekdf[weekdf['difference'] > 0])
    pct_win_week = (win_trades_week / total_trades_week) * 100 if total_trades_week > 0 else 0
    sum_week_usdt = sum_week * 0.001
    # dia
    day_back = datetime.now() - timedelta(hours=24)
    daydf = df[df['date'] >= day_back]
    sum_day = round(daydf['difference'].sum(),2)
    total_trades_day = len(daydf)
    win_trades_day = len(daydf[daydf['difference'] > 0])
    pct_win_day = (win_trades_day / total_trades_day) * 100 if total_trades_day > 0 else 0
    sum_day_usdt = sum_day * 0.001

    headers_total = ["Num. Operaciones Total","Suma Operaciones Total",  "% Ganadoras Total", "USDT Ganado Total"]
    headers_month = ["Num. Operaciones Mes","Suma Operaciones Mes",  "% Ganadoras Mes", "USDT Ganado Mes"]
    headers_week = ["Num. Operaciones Semana","Suma Operaciones Semana",  "% Ganadoras Semana", "USDT Ganado Semana"]
    headers_day = ["Num. Operaciones Dia","Suma Operaciones Dia",  "% Ganadoras Dia", "USDT Ganado Dia"]

    data_total = [total_trades,total_sum,f"{pct_win:.2f}%", f"{total_sum_usdt:.2f}"]
    data_month = [total_trades_month,sum_month,  f"{pct_win_month:.2f}%", f"{sum_month_usdt:.2f}"]
    data_week = [total_trades_week,sum_week,  f"{pct_win_week:.2f}%", f"{sum_week_usdt:.2f}"]
    data_day = [total_trades_day,sum_day,  f"{pct_win_day:.2f}%", f"{sum_day_usdt:.2f}"]

    log_write_to_csv('../data/csv/balance_statistics_total.csv', headers_total, data_total)
    log_write_to_csv('../data/csv/balance_statistics_month.csv', headers_month, data_month)
    log_write_to_csv('../data/csv/balance_statistics_week.csv', headers_week, data_week)
    log_write_to_csv('../data/csv/balance_statistics_day.csv', headers_day, data_day)
  

def main():
    balanceStatistics()
    return 0

if __name__ == "__main__":
    main()
