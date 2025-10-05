# Tesla Stock Price Forecasting — LSTM + Flask

**Project summary**
This project implements a time-series forecasting pipeline using an LSTM neural network and exposes it via a lightweight Flask web application for interactive forecasts of stock closing prices (example: TSLA). The app accepts different tickers and uses a pre-trained LSTM model to produce a multi-day business-day forecast.

**Files provided**
- `model.ipynb` — Jupyter notebook that trains the LSTM model.
- `app.py` — Flask application that loads the trained model and serves forecasts. (See the app code for details.)

**Evaluation**
- MAE: **16.30**
- RMSE: **21.29**
- R²: **0.859**

These metrics indicate the model explains approximately 86% of the variance on the test set, with an average absolute error of ~16 price units, demonstrating strong predictive accuracy.

---
##🖥️ Live Demo
![image alt](https://github.com/Raiyyan355/Time-series-forecasting--LSTM-/blob/main/screenshot-1759143922485.png?raw=true)


## Project steps (how it works — end-to-end)

1. **Data ingestion**
   - Historical OHLCV data is downloaded via `yfinance` (example code in notebook). The app also fetches fresh data for any ticker at runtime.

2. **Preprocessing**
   - The notebook extracts the `Close` price series and scales it (`MinMaxScaler`) before creating supervised sequences using a sliding window (default: 60 past days → next day).

3. **Train / validation / test split**
   - The notebook uses a chronological split (80% train / 20% test). Training avoids shuffling to prevent lookahead bias.

4. **Model architecture**
   - A 2-layer LSTM network (example: 100 units per layer) with dropout and Dense head is trained using MSE loss and Adam optimizer. The notebook computes MAE, RMSE, and R² on the held-out test set.

5. **Saving model & scaler**
   - After training, save the model in Keras format so the Flask app can load it:
     ```python
     model.save('models/LSTM_model.keras')   # or 'models/saved_model.h5'
     ```
   - Save the scaler used during preprocessing (highly recommended):
     ```python
     from joblib import dump
     dump(scaler, 'models/scaler.joblib')
     ```

6. **Flask app (runtime)**
   - `app.py` loads the saved Keras model (path configured inside the file) and, for a requested ticker:
     - fetches recent `Close` prices via `yfinance`
     - scales the series using a fresh `MinMaxScaler()` or the saved scaler (if you saved one)
     - uses the last `window` scaled observations to iteratively predict `n` future business days
     - inverse-transforms predictions and returns an interactive plot on a result page
   - See `app.py` for implementation details. fileciteturn0file0

---

## How to run (local demo)

1. Place trained model and scaler into the `models/` folder:
   - `models/LSTM_model.keras` (or `models/saved_model.h5`)
   - `models/scaler.joblib` (optional but recommended)

2. Install requirements (use a venv):
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows (PowerShell)
   pip install -r requirements.txt
   ```

3. Start the Flask app:
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

---

## How to save the correct model from the notebook (critical)

If you previously saved the Keras training `history` to disk by mistake (e.g., `joblib.dump(history, ...)`), that will cause the app to fail because the `History` object has no `.predict()` method.

**Correct saving — Keras model + scaler**
```python
# After training
model.fit(x_train, y_train, epochs=E, batch_size=B, validation_data=(x_val, y_val))
# Save model (Keras format)
model.save('models/LSTM_model.keras')   # or .h5

# Save scaler so Flask uses same scaling
from joblib import dump
dump(scaler, 'models/scaler.joblib')
```

**If you only have a joblib file that contains a History object**, you must retrain or re-export the model correctly.

## 📁 Project Structure

```
├── app.py                  # Flask web app
├── models
|    └──LSTM_model.keras    # Pre-trained ML model (to be added)
├── templates/
│   └── index.html,result.html                  
├── README.md               # Project documentation
├── model.ipynb             # notebok           
└── requirements.txt        # Dependencies list
```


✅ **This project demonstrates practical time series forecasting techniques applied to real-world financial data with clear visualization and performance evaluation.**
