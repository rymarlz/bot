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
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import mplfinance as mpf

app = Flask(__name__)
socketio = SocketIO(app)

# Configuración del cliente de Binance
api_key = config.API_KEY
api_secret = config.API_SECRET
client = Client(api_key, api_secret)

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)

# Variables para las fechas de inicio y fin en formato DD/MM/YYYY HH:MM:SS
start_date = "01/01/2023 00:00:00"
end_date = "24/07/2024 23:00:00"

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
    """Obtiene datos históricos de Binance y los retorna en un DataFrame."""
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
    """Filtra los símbolos que cumplen con la condición de EMA200."""
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
    """Identifica reversiones de tendencia en los datos utilizando regresión polinomial ajustada."""
    if len(data) < 60:
        return False, data

    log_message("Calculando línea de tendencia con regresión polinomial ajustada")

    data['index'] = np.arange(len(data))

    # Crear características polinomiales
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(data[['index']])
    
    # Ajustar el modelo de regresión lineal
    model = LinearRegression()
    model.fit(X_poly, data['close'])
    
    # Predecir la línea de tendencia
    data['trendline'] = model.predict(X_poly)
    
    # Ajuste adicional para elevar la línea de tendencia
    adjustment_factor = (data['close'].max() - data['trendline'].max()) * 0.2
    data['trendline'] += adjustment_factor
    
    log_message(f"Línea de tendencia ajustada con un factor de ajuste de {adjustment_factor}")

    # Identificar si el precio ha cruzado por encima o por debajo de la línea de tendencia
    data['above_trend'] = data['close'] > data['trendline']
    data['below_trend'] = data['close'] < data['trendline']

    crossed_above = data['above_trend'].diff().eq(1).any()
    crossed_below = data['below_trend'].diff().eq(1).any()

    last_close = data['close'].iloc[-1]
    last_trend = data['trendline'].iloc[-1]

    log_message(f"Último cierre: {last_close}, Última línea de tendencia: {last_trend}")

    if (crossed_above or crossed_below) and last_close <= last_trend:
        if last_close <= last_trend:
            return True, data

    return False, data

def apply_fibonacci(data):
    """Aplica niveles de Fibonacci a los datos y retorna los puntos usados."""
    data = data[-60:].copy()
    max_price = data['high'].max()
    min_price = data['low'].min()
    diff = max_price - min_price
    levels = {
        'Fibo_0.618': max_price - diff * 0.618,
        'Fibo_0.705': max_price - diff * 0.705,
        'Fibo_0.785': max_price - diff * 0.785
    }
    for level_name, level_value in levels.items():
        data[level_name] = level_value
    return data, max_price, min_price


def save_to_excel(symbol, data, writer):
    """Guarda los datos en un archivo Excel."""
    data.tail(60).to_excel(writer, sheet_name=symbol)  # Guardar solo los últimos 60 datos

def create_candlestick_chart(symbol, data, max_price, min_price):
    """Crea y guarda un gráfico de velas con niveles de Fibonacci."""
    output_folder = 'static/charts'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f'{symbol}.png')

    data_to_plot = data.tail(100).copy()
    data_to_plot.index.name = 'Date'

    trend_data = data.tail(60).copy()
    date_nums = mdates.date2num(trend_data.index)
    slope, intercept, _, _, _ = linregress(date_nums, trend_data['close'])
    trend_data['trendline'] = intercept + slope * date_nums

    log_message(f"date_nums: {date_nums}")
    log_message(f"slope: {slope}, intercept: {intercept}")
    log_message(f"trendline: {trend_data['trendline'].values}")

    # Dibujar niveles de Fibonacci
    fibo_levels = ['Fibo_0.618', 'Fibo_0.705', 'Fibo_0.785']
    fibo_plots = [mpf.make_addplot(data[level], color='purple', linestyle='--') for level in fibo_levels if level in data.columns]
    
    # Marcar puntos de Fibonacci
    data_to_plot['max_point'] = np.nan
    data_to_plot['min_point'] = np.nan
    data_to_plot.loc[data.index[data['high'] == max_price][0], 'max_point'] = max_price
    data_to_plot.loc[data.index[data['low'] == min_price][0], 'min_point'] = min_price
    fibo_plots.append(mpf.make_addplot(data_to_plot['max_point'], color='red', marker='^', markersize=10))
    fibo_plots.append(mpf.make_addplot(data_to_plot['min_point'], color='green', marker='v', markersize=10))
    
    # Crear el gráfico de velas usando mplfinance
    ap = [mpf.make_addplot(trend_data['trendline'], color='blue', panel=0)] + fibo_plots
    mpf.plot(data_to_plot, type='candle', style='charles', addplot=ap, volume=True, 
             title=f'Gráfico de Velas para {symbol} con Línea de Tendencia y Fibonacci', 
             savefig=dict(fname=output_file, dpi=100, bbox_inches='tight', pad_inches=0.1))

    log_message(f"Gráfico de velas con línea de tendencia y Fibonacci para {symbol} guardado en {output_file}")

def run_strategy():
    """Ejecuta la estrategia de análisis."""
    log_message("Conectado a Binance")
    symbols = [symbol['symbol'] for symbol in client.get_all_tickers() if symbol['symbol'].endswith('USDT')]
    log_message(f"Total de símbolos con par USDT: {len(symbols)}")

    output_folder = 'data'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, 'filtro_simbolos.xlsx')

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
                        data_15m, max_price, min_price = apply_fibonacci(data_15m)
                        results.append(symbol)
                        data_to_save = pd.concat([data_4h[['EMA_200']], data_15m], axis=1)
                        save_to_excel(symbol, data_to_save, writer)
                        create_candlestick_chart(symbol, data_15m, max_price, min_price)
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
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=True)

