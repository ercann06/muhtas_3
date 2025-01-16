import yfinance as yf
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import Holt
from datetime import timedelta
import tkinter as tk
from tkinter import ttk
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Apple hisse fiyatlarını yfinance ile çekme (son 1 yıl)
ticker = 'AAPL'
data = yf.download(ticker, period="1y", interval="1d")
df = data[['Close']].reset_index()
df.columns = ['ds', 'y']

# Tarih sütununu datetime formatına dönüştürme
df['ds'] = pd.to_datetime(df['ds'])

# Son 10 veriyi tutma
forecast_steps = 10
train_data = df.iloc[:-forecast_steps]

# Prophet Modeli
prophet = Prophet()
prophet.fit(train_data)
future = prophet.make_future_dataframe(periods=forecast_steps)
prophet_forecast = prophet.predict(future)[-forecast_steps:]["yhat"].values

# ARIMA Modeli
arima_model = ARIMA(train_data["y"], order=(1, 1, 1))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=forecast_steps).values

# ETS Modeli
ets_model = ExponentialSmoothing(train_data["y"], seasonal=None, trend="add", damped_trend=True)
ets_fit = ets_model.fit()
ets_forecast = ets_fit.forecast(steps=forecast_steps).values

# Holt Modeli
holt_model = Holt(train_data["y"], damped_trend=True)
holt_fit = holt_model.fit()
holt_forecast = holt_fit.forecast(steps=forecast_steps).values

# XGBoost Modeli
lags = 10
xgb_df = pd.DataFrame({f"lag_{i}": train_data["y"].shift(i) for i in range(1, lags + 1)})
xgb_df["target"] = train_data["y"]
xgb_df.dropna(inplace=True)

X_train = xgb_df.iloc[:-forecast_steps, :-1]
y_train = xgb_df.iloc[:-forecast_steps, -1]
X_test = xgb_df.iloc[-forecast_steps:, :-1]

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_forecast = xgb_model.predict(X_test)

# Gerçek veriler
real_data = df.iloc[-forecast_steps:]["y"].values

# Örnek tahmin sonuçları
forecast_dates = [df["ds"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_steps + 1)]
results = pd.DataFrame({
    "Date": forecast_dates,
    "Prophet": prophet_forecast,
    "ARIMA": arima_forecast,
    "ETS": ets_forecast,
    "Holt": holt_forecast,
    "XGBoost": xgb_forecast,
    "Real Data": real_data
})

results["Date"] = results["Date"].dt.strftime('%d.%m.%y')


# GUI oluşturma
def display_table_and_graph():
    root = tk.Tk()
    root.title("Forecast Results")

    # Frame oluşturma
    frame = tk.Frame(root)
    frame.pack(expand=True, fill=tk.BOTH)

    # Treeview oluşturma
    tree = ttk.Treeview(frame, columns=list(results.columns), show="headings")
    for col in results.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Tabloya verileri ekleme
    for _, row in results.iterrows():
        tree.insert("", tk.END, values=list(row))

    tree.grid(row=0, column=0, padx=10, pady=10)

    # Grafik oluşturma
    fig, ax = plt.subplots(figsize=(8, 6))
    for col in results.columns[1:-1]:  # Gerçek verileri hariç tut
        ax.plot(results["Date"], results[col], label=col)

    # Gerçek veriyi ekleme
    ax.plot(results["Date"], results["Real Data"], label="Real Data", color='black', linestyle='--', marker='o')

    ax.set_title("Apple Stock Price Forecast Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecasted Value")
    ax.legend()

    # Grafik Tkinter penceresine yerleştirilmesi
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)

    # Çıkış butonu
    quit_button = tk.Button(root, text="Exit", command=root.destroy)
    quit_button.pack()

    root.mainloop()


# GUI'yi başlat
display_table_and_graph()
