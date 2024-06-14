from flask import Flask, jsonify, render_template, request
import pandas as pd
import subprocess
#import ssl

#context = ssl.SSLContext()
#context.load_cert_chain('cert.pem', 'key.pem')

#app = Flask(__name__)
app = Flask(__name__, template_folder='../frontend/templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/modelos')
def services():
    return render_template('modelos.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/data')
def data():
    return render_template('data.html')

@app.route('/datacsv')
def get_data():
    df = pd.read_csv('../src/log_all_orders.csv')
    columns = df.columns.tolist()
    last_20 = df.tail(20).to_dict(orient='records')
    return jsonify({'columns': columns, 'data': last_20})

@app.route('/validate')
def validate():
    return render_template('validate.html')

@app.route('/validatecsv')
def get_validate():
    df = pd.read_csv('../src/xxxvalidatepredicts.csv')
    df = df.sort_values(by='predict', ascending=False)
    last_20 = df.to_dict(orient='records')
    columns = df.columns.tolist()
    return jsonify({'columns': columns, 'data': last_20})

@app.route('/runbot', methods=['POST'])
def runbot():
    # Bot params
    interval = request.form.get('interval', '5m')
    live = request.form.get('live', 'True')
    numrecords = request.form.get('numrecords', '289')

    # Script con params
    try:
        result = subprocess.run(['python', 'C:\\BotTestnetv4.py', interval, live, numrecords], capture_output=True, text=True)
        return jsonify({'status': 'success', 'output': result.stdout})
    except Exception as e:
        return jsonify({'status': 'error', 'output': str(e)})

if __name__ == '__main__':
    #app.run(host='127.0.0.1',port=443, debug=True, ssl_context=context)
    #app.run(debug=True, ssl_context=("cert.pem", "key.pem"))
    app.run(host='127.0.0.1', debug=True)
