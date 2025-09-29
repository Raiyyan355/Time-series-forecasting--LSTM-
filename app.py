from flask import Flask, render_template, request, redirect, url_for, flash
import io, base64, os, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Path to the saved joblib model
MODEL_PATH = os.path.join(os.getcwd(), 'models', 'LSTM_model.keras')

model = load_model(MODEL_PATH)
  

def fetch_close_series(ticker, start='2020-01-01', end=None):
    end = end or datetime.datetime.today().strftime('%Y-%m-%d')
    try:
        df = yf.download(ticker, start=start, end=end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df[['Close']]

def naive_forecast(df, days=30):
    """Fallback forecast if model is missing."""
    last = float(df['Close'].iloc[-1])
    dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    return pd.DataFrame({'Prediction': [last]*days}, index=dates)

def forecast_with_model(model, data_df, window=60, days=30):
    """Forecast future prices using the trained LSTM model."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data_df)
    last_window = scaled[-window:]
    x_future = last_window.reshape((1, last_window.shape[0], 1))

    preds = []
    for _ in range(days):
        pred = model.predict(x_future)
        preds.append(pred[0, 0])
        x_future = np.concatenate([x_future[:, 1:, :], pred.reshape(1, 1, 1)], axis=1)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    dates = pd.date_range(start=data_df.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
    return pd.DataFrame({'Prediction': preds}, index=dates)

def plot_history_and_forecast(history_df, forecast_df, ticker):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history_df.index, history_df['Close'], label='Historical Close')
    ax.plot(forecast_df.index, forecast_df['Prediction'], linestyle='--', label='Forecast')
    ax.set_title(f'{ticker} Close Price Forecast')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    fig.tight_layout()

    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    plt.close(fig)
    buf.seek(0)
    return buf

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    ticker = request.form.get('ticker', 'TSLA').upper().strip()
    window = int(request.form.get('window', 60))
    days = int(request.form.get('days', 30))
    start = request.form.get('start', '2020-01-01')

    df = fetch_close_series(ticker, start=start)
    if df is None or df.empty:
        flash('Could not fetch data for ticker. Check the symbol or internet connection.', 'danger')
        return redirect(url_for('index'))

    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            flash(f'Failed to load model: {e}', 'warning')

    if model is not None:
        forecast_df = forecast_with_model(model, df, window=window, days=days)
    else:
        # fallback forecast silently
        forecast_df = naive_forecast(df, days=days)

    plot_buf = plot_history_and_forecast(df, forecast_df, ticker)
    plot_base64 = base64.b64encode(plot_buf.getvalue()).decode('utf-8')

    last_close = float(df['Close'].iloc[-1])
    first_pred = float(forecast_df['Prediction'].iloc[0])
    last_pred = float(forecast_df['Prediction'].iloc[-1])

    return render_template(
        'result.html',
        ticker=ticker,
        last_close=last_close,
        first_pred=first_pred,
        last_pred=last_pred,
        plot_base64=plot_base64
    )

if __name__ == "__main__":
    app.run(debug=True)
