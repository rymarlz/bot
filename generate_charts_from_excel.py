import pandas as pd
import mplfinance as mpf
import os

def log_message(message):
  print(message)

def heikin_ashi(df):
  ha_df = df.copy()

  ha_df['ha_close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

  ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
  for i in range(1, len(df)):
    ha_open.append((ha_open[i-1] + ha_df['ha_close'].iloc[i-1]) / 2)

  ha_df['ha_open'] = ha_open
  ha_df['ha_high'] = ha_df[['high', 'ha_open', 'ha_close']].max(axis=1)
  ha_df['ha_low'] = ha_df[['low', 'ha_open', 'ha_close']].min(axis=1)

  return ha_df[['ha_open', 'ha_high', 'ha_low', 'ha_close', 'volume']]

def generate_heikin_ashi_charts(excel_file, log_func=log_message, output_folder='heikin_ashi_charts'):
  # Cargar el archivo Excel
  try:
    xls = pd.ExcelFile(excel_file)
    log_func(f"Archivo Excel {excel_file} cargado correctamente")
  except Exception as e:
    log_func(f"Error al cargar el archivo Excel: {e}")
    return

  # Crear una carpeta para los gráficos de velas Heikin-Ashi
  try:
    if not os.path.exists(output_folder):
      os.makedirs(output_folder)
    log_func(f"Carpeta para gráficos de velas Heikin-Ashi: {output_folder}")
  except Exception as e:
    log_func(f"Error al crear la carpeta para gráficos de velas Heikin-Ashi: {e}")
    return

  # Iterar sobre cada hoja (símbolo) en el archivo Excel
  for sheet_name in xls.sheet_names:
    log_func(f"Procesando hoja: {sheet_name}")
    try:
      df = pd.read_excel(xls, sheet_name=sheet_name, index_col=0, parse_dates=True)
      df.index.name = 'timestamp'
      df.columns = df.columns.str.strip().str.lower() # Limpiar y pasar los nombres de las columnas a minúsculas
      log_func(f"Nombres de columnas de la hoja {sheet_name}: {df.columns.tolist()}")
    except Exception as e:
      log_func(f"Error al leer la hoja {sheet_name}: {e}")
      continue

    # Verificar que las columnas necesarias existan
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
            log_func(f"Columnas faltantes en la hoja {sheet_name}: {missing_columns}")
            continue

    # Calcular velas Heikin-Ashi
    try:
      ha_df = heikin_ashi(df)
      mpf.plot(ha_df, type='candle', volume=True, style='charles',
          title=f'Gráfico de Velas Heikin-Ashi de {sheet_name}',
          ylabel='Precio', ylabel_lower='Volumen',
          savefig=os.path.join(output_folder, f'{sheet_name}.png'))
      log_func(f"Gráfico de velas Heikin-Ashi guardado para {sheet_name}")
    except Exception as e:
      log_func(f"Error al generar o guardar el gráfico de velas Heikin-Ashi para {sheet_name}: {e}")

  log_func("Generación de gráficos de velas Heikin-Ashi completada")

if __name__ == '__main__':
  # Establecer la ruta del archivo Excel
  excel_file = 'filtro_simbolos.xlsx'  # Ajustar la ruta del archivo
  log_message("Iniciando generación de gráficos de velas Heikin-Ashi...")
  generate_heikin_ashi_charts(excel_file)
  log_message("Finalizando generación de gráficos de velas Heikin-Ashi...")
