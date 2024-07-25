import eventlet
eventlet.monkey_patch()

from flask import Flask, jsonify, render_template
from flask_socketio import SocketIO, emit
import os
import pandas as pd
import numpy as np
from binance.client import Client
from scipy.stats import linregress
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import logging
import matplotlib.pyplot as plt
import mplfinance as mpf
import seaborn as sns
import matplotlib.dates as mdates

app = Flask(__name__)
socketio = SocketIO(app)

# Configuración del cliente de Binance
api_key = config.API_KEY
api_secret = config.API_SECRET
client = Client(api_key, api_secret)

logging.basicConfig(level=logging.DEBUG)

# Variables para las fechas de inicio y fin en formato DD/MM/YYYY HH:MM:SS
start_date = "01/01/2023 00:00:00"
end_date = "16/07/2024 00:00:00"

# Convertir las fechas al formato que espera la API de Binance
start_date_binance = pd.to_datetime(start_date, format='%d/%m/%Y %H:%M:%S').strftime('%d %b, %Y %H:%M:%S')
end_date_binance = pd.to_datetime(end_date, format='%d/%m/%Y %H:%M:%S').strftime('%d %b, %Y %H:%M:%S')

# Funciones auxiliares
def log_message(message):
    logging.debug(message)
    socketio.emit('log', {'data': message})

def emit_progress(progress):
    socketio.emit('progress', {'progress': progress})

def get_historical_data(symbol, interval):
    log_message(f"Obteniendo datos históricos para {symbol} con intervalo {interval}")
    klines = client.get_historical_klines(symbol, interval, start_date_binance, end_date_binance)
    data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'number_of_trades', 
                                         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    data.set_index('timestamp', inplace=True)
    data = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    log_message(f"Datos históricos para {symbol} obtenidos correctamente")
    return data

def filter_symbol_by_ema200(symbol):
    try:
        data_4h = get_historical_data(symbol, '4h')
        if len(data_4h) >= 200:  # Verificar si hay suficientes datos para calcular la EMA200
            data_4h['EMA_200'] = data_4h['close'].ewm(span=200).mean()
            latest_close = data_4h['close'].iloc[-1]
            latest_ema200 = data_4h['EMA_200'].iloc[-1]
            if latest_close >= latest_ema200 and latest_close <= latest_ema200 * 1.03:
                log_message(f"{symbol} cumple con el filtro EMA200")
                return symbol, data_4h
        else:
            log_message(f"{symbol} no tiene suficientes datos para calcular la EMA200")
    except Exception as e:
        log_message(f"Error procesando el símbolo {symbol} para filtro EMA200: {e}")
    return None, None

def find_trend_reversal(data):
    if len(data) < 60:
        return False, data
    log_message("Calculando línea de tendencia")
    data['index'] = np.arange(len(data))
    slope, intercept, _, _, _ = linregress(data['index'], data['close'])
    data['trendline'] = intercept + slope * data['index']
    log_message(f"Línea de tendencia calculada: Slope={slope}, Intercept={intercept}")

    return True, data

def apply_fibonacci(data):
    data = data[-60:].copy()
    max_price = data['high'].max()
    min_price = data['low'].min()
    diff = max_price - min_price
    levels = [max_price - diff * ratio for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]]
    for level in levels:
        data.loc[:, f'Fibo_{level:.3f}'] = level
    return data

def save_to_excel(symbol, data, writer):
    data.tail(60).to_excel(writer, sheet_name=symbol)  # Guardar solo los últimos 60 datos


def create_candlestick_chart(symbol, data):
    output_folder = 'static/charts'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f'{symbol}.png')

    # Seleccionar los últimos 60 datos para el gráfico
    data_to_plot = data.tail(60).copy()
    data_to_plot.index.name = 'Date'

    # Calcular la línea de tendencia usando regresión lineal en los datos de tiempo
    date_nums = mdates.date2num(data_to_plot.index)
    slope, intercept, _, _, _ = linregress(date_nums, data_to_plot['close'])
    data_to_plot['trendline'] = intercept + slope * date_nums

    # Mensajes de depuración
    log_message(f"date_nums: {date_nums}")
    log_message(f"slope: {slope}, intercept: {intercept}")
    log_message(f"trendline: {data_to_plot['trendline'].values}")

    # Crear el gráfico de velas usando mplfinance
    ap = [mpf.make_addplot(data_to_plot['trendline'], color='blue', panel=0)]
    mpf.plot(data_to_plot, type='candle', style='charles', addplot=ap, volume=True, title=f'Gráfico de Velas para {symbol} con Línea de Tendencia', savefig=dict(fname=output_file, dpi=100, bbox_inches='tight', pad_inches=0.1))

    log_message(f"Gráfico de velas con línea de tendencia para {symbol} guardado en {output_file}")


def run_strategy():
    log_message("Conectado a Binance")
    # Obtener todos los símbolos con el par USDT
    symbols = [symbol['symbol'] for symbol in client.get_all_tickers() if symbol['symbol'].endswith('USDT')]
    log_message(f"Total de símbolos con par USDT: {len(symbols)}")

    output_folder = 'data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, 'filtro_simbolos.xlsx')

    # Filtrar los símbolos que cumplen con la condición de la EMA200 usando hilos
    filtered_symbols = []
    total_symbols = len(symbols)
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(filter_symbol_by_ema200, symbol): symbol for symbol in symbols}
        for i, future in enumerate(as_completed(futures)):
            symbol, data_4h = future.result()
            if symbol is not None and data_4h is not None:
                filtered_symbols.append((symbol, data_4h))
            progress = (i + 1) / total_symbols * 50
            emit_progress(progress)

    # Crear un writer de Excel para guardar los datos
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        results = []
        total_filtered_symbols = len(filtered_symbols)
        futures = {}
        with ThreadPoolExecutor(max_workers=20) as executor:
            for symbol, data_4h in filtered_symbols:
                futures[executor.submit(get_historical_data, symbol, '15m')] = (symbol, data_4h)
            for i, future in enumerate(as_completed(futures)):
                symbol, data_4h = futures[future]
                try:
                    data_15m = future.result()
                    trend_reversal_detected, data_15m = find_trend_reversal(data_15m)
                    if trend_reversal_detected:
                        data_15m = apply_fibonacci(data_15m)
                        results.append(symbol)
                        data_to_save = pd.concat([data_4h[['EMA_200']], data_15m], axis=1)
                        save_to_excel(symbol, data_to_save, writer)
                        create_candlestick_chart(symbol, data_15m)
                except Exception as e:
                    log_message(f"Error procesando el símbolo {symbol} para reversión de tendencia: {e}")
                progress = 50 + (i + 1) / total_filtered_symbols * 50
                emit_progress(progress)

    log_message("Símbolos con configuración de reversión de tendencia:")
    for symbol in results:
        log_message(symbol)
    socketio.emit('results', {'symbols': results})
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    threading.Thread(target=run_strategy).start()
    return jsonify({'status': 'Estrategia iniciada'})

if __name__ == '__main__':
    socketio.run(app, debug=True)