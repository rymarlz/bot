import pandas as pd
import numpy as np
from binance.client import Client
import datetime

def get_binance_client(api_key, api_secret):
    print("Inicializando cliente de Binance")
    return Client(api_key, api_secret)

def get_futures_symbols(client):
    print("Obteniendo símbolos de futuros con par USDT")
    exchange_info = client.futures_exchange_info()
    symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['quoteAsset'] == 'USDT']
    print(f"Símbolos obtenidos: {symbols}")
    return symbols

def get_historical_data(client, symbol, interval, limit=60):
    print(f"Descargando datos históricos para {symbol} con intervalo {interval}")
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    data = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                                         'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 
                                         'taker_buy_quote_asset_volume', 'ignore'])
    data['open_time'] = pd.to_datetime(data['open_time'], unit='ms')
    data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')
    data[['open', 'high', 'low', 'close', 'volume']] = data[['open', 'high', 'low', 'close', 'volume']].astype(float)
    print(f"Datos históricos descargados para {symbol}")
    return data

def calculate_ema(data, period=200):
    print("Calculando EMA")
    ema = data['close'].ewm(span=period, adjust=False).mean()
    print(f"EMA calculada: {ema.iloc[-5:]}")  # Imprimir últimas 5 EMA para debug
    return ema

def calculate_trendline(data):
    print("Calculando línea de tendencia")
    x = np.arange(len(data))
    y = data['close'].values
    fit = np.polyfit(x, y, 1)
    trend = np.poly1d(fit)
    print(f"Línea de tendencia calculada: {trend.coefficients}")
    return trend(x)

def filter_symbols(client):
    symbols = get_futures_symbols(client)
    filtered_symbols = []
    for idx, symbol in enumerate(symbols):
        try:
            print(f"Procesando símbolo {symbol} ({idx+1}/{len(symbols)})")
            daily_data = get_historical_data(client, symbol, '1d', 200)
            ticker_info = client.futures_symbol_ticker(symbol=symbol)
            
            if 'price' not in ticker_info:
                print(f"Error: La respuesta de la API no contiene 'price' para el símbolo {symbol}")
                continue
            
            current_price = float(ticker_info['price'])
            daily_ema = calculate_ema(daily_data)
            
            if daily_ema.iloc[-1] <= current_price <= daily_ema.iloc[-1] * 1.03:
                minute_data = get_historical_data(client, symbol, '15m', 60)
                trendline = calculate_trendline(minute_data)
                under_trendline = minute_data['close'] < trendline
                if under_trendline.any() and not under_trendline.all():
                    if current_price < trendline[-1]:
                        filtered_symbols.append(symbol)
                        print(f"Símbolo {symbol} agregado a la lista de filtrados")
            
            print(f"Progreso: {((idx + 1) / len(symbols)) * 100:.2f}%")
        
        except Exception as e:
            print(f"Error procesando el símbolo {symbol}: {e}")

    print(f"Símbolos filtrados: {filtered_symbols}")
    return filtered_symbols
