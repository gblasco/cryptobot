import os
from flask import Flask, jsonify, render_template, request, redirect, url_for, session, flash
import pandas as pd
import subprocess
import json
from functools import wraps

app = Flask(__name__)
app.secret_key = os.urandom(42)

gbbpassword = os.getenv('FRONT_USER1_KEY')
adminpassword = os.getenv('FRONT_USER2_KEY')

users = {
    'gbb': gbbpassword,
    'admin': adminpassword
}

def mandatory_login(f):
    @wraps(f)
    def wrapper_login(*args, **kwargs):
        if 'username' not in session:
            flash('Logeate para poder ver esta pagina!', 'info')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper_login

@app.route('/main')
@mandatory_login
def index():
    return render_template('main.html')

@app.route('/about')
@mandatory_login
def about():
    return render_template('about.html')

@app.route('/sentiments')
@mandatory_login
def sentiments():
    result = subprocess.run(['python', 'sentimentsAnalysis.py'], capture_output=True, text=True)
    df = pd.read_csv('../data/csv/sentiments_bitcoin.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return render_template('sentiments.html', sentiment_result=result.stdout, headers=headers, data=data)

@app.route('/run-sentiment-analysis')
@mandatory_login
def run_sentiment_analysis():
    result = subprocess.run(['python', 'sentimentsAnalysis.py'], capture_output=True, text=True)
    return jsonify(result=result.stdout, image_generated=os.path.exists('static/sentimentsplot.png'))

@app.route('/sentimentscsv')
@mandatory_login
def get_sentiments():
    df = pd.read_csv('../data/csv/sentiments_bitcoin.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return jsonify({'headers': headers, 'data': data})

@app.route('/wallet')
@mandatory_login
def wallet():
    result = subprocess.run(['python', 'wallet.py'], capture_output=True, text=True)
    return render_template('wallet.html', wallet_result=result.stdout)

@app.route('/walletcsv')
@mandatory_login
def get_wallet():
    column_names = ['btc_balance', 'btc_usdt', 'usdt_balance', 'usdt_balance2', 'btc_price', 'total_balance_usd']
    df = pd.read_csv('../data/csv/wallet.csv', header=None, names=column_names)
    data = [
        {'Crypto': 'BTC', 'Balance': df['btc_balance'].iloc[0], 'SecondaryBalance': df['btc_usdt'].iloc[0]},
        {'Crypto': 'USDT', 'Balance': df['usdt_balance'].iloc[0], 'SecondaryBalance': df['usdt_balance2'].iloc[0]}
    ]
    columns = ['Crypto', 'Balance']
    total_balance_usd = df['total_balance_usd'].iloc[0]
    return jsonify({'columns': columns, 'data': data, 'total_balance_usd': total_balance_usd})

@app.route('/data')
@mandatory_login
def data():
    return render_template('data.html')

@app.route('/datacsv')
@mandatory_login
def get_data():
    df = pd.read_csv('../src/log_all_orders.csv')
    columns = df.columns.tolist()
    last_100 = df.tail(100).to_dict(orient='records')
    return jsonify({'columns': columns, 'data': last_100})

# @app.route('/predicts')
# def validate():
#     return render_template('predicts.html')

# @app.route('/predictscsv')
# def get_predicts():
#     df = pd.read_csv('../src/log_all_orders.csv')
#     data = df.to_dict(orient='records')
#     columns = df.columns.tolist()
#     return jsonify(columns=columns, data=data)

@app.route('/trade')
@mandatory_login
def trade():
    result = subprocess.run(['python', 'balanceStatistics.py'], capture_output=True, text=True)
    return render_template('trade.html', statistics_result=result.stdout)

@app.route('/tradecsvtotal')
@mandatory_login
def get_trade_total():
    df = pd.read_csv('../data/csv/balance_statistics_total.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return jsonify({'headers': headers, 'data': data})

@app.route('/tradecsvmonth')
@mandatory_login
def get_trade_month():
    df = pd.read_csv('../data/csv/balance_statistics_month.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return jsonify({'headers': headers, 'data': data})

@app.route('/tradecsvweek')
@mandatory_login
def get_trade_week():
    df = pd.read_csv('../data/csv/balance_statistics_week.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return jsonify({'headers': headers, 'data': data})

@app.route('/tradecsvday')
@mandatory_login
def get_trade_day():
    df = pd.read_csv('../data/csv/balance_statistics_day.csv', header=0)
    headers = list(df.columns)
    data = df.to_dict(orient='records')
    return jsonify({'headers': headers, 'data': data})

@app.route('/tradecsv')
@mandatory_login
def get_trade():
    df = pd.read_csv('../src/log_buysell_orders.csv', header=None, names=['timestamp', 'price1', 'price2', 'difference'])
    data = df.to_dict(orient='records')
    return jsonify({'data': data})

@app.route('/runbot', methods=['POST'])
@mandatory_login
def runbot():
    # Bot params
    interval = request.form.get('interval', '5m')
    live = request.form.get('live', 'True')
    numrecords = request.form.get('numrecords', '289')

    # Script con params
    try:
        result = subprocess.run(['python', '..\src\BotCryptobotv5.py', interval, live, numrecords], capture_output=True, text=True)
        return jsonify({'status': 'success', 'output': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)})

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            flash('Login correcto!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Nombre de usuario o clave incorrectos!', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Te has deslogeado con exito!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
