import tkinter as tk
from tkinter import ttk, messagebox
import sqlite3
import yfinance as yf
from datetime import datetime
from yahooquery import search 
import csv
from tkcalendar import DateEntry

# Deep Learning & Analysis Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


DB_NAME = "financial_data.db"

def get_ticker_from_name(company_name: str):

    results = search(company_name)
    quotes = results.get("quotes", [])
    if quotes:
        return quotes[0]["symbol"]  # Grab the first matching ticker
    return None



def create_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
 
    cursor.execute("DROP TABLE IF EXISTS stock_data")
    
  
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            frequency TEXT
        )
    ''')
    conn.commit()
    conn.close()


def fetch_and_store_data():
  
    ticker = ticker_var.get()
    start_date = from_date_var.get()
    end_date = to_date_var.get()
    frequency = freq_var.get() 
    if not ticker or not start_date or not end_date or not frequency:
        messagebox.showerror("Input Error", "Please fill all fields and select a frequency.")
        return

    try:
        log_message(f"Fetching {frequency} data for {ticker} from {start_date} to {end_date}...")

    
        start_date = datetime.strptime(start_date, "%Y-%m-%d").strftime("%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d").strftime("%Y-%m-%d")

     
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=frequency)

        if stock_data.empty:
            messagebox.showerror("No Data", "No data available for the given period/frequency.")
            return

        # Store data in SQLite
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        for date, row in stock_data.iterrows():
            cursor.execute('''
                INSERT INTO stock_data (ticker, date, open, high, low, close, volume, frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ticker,
                date.strftime("%Y-%m-%d"),
                float(row['Open'].iloc[0]),
                float(row['High'].iloc[0]),
                float(row['Low'].iloc[0]),
                float(row['Close'].iloc[0]),
                int(row['Volume'].iloc[0]),
                frequency
            ))

        conn.commit()
        conn.close()

        log_message("Data fetched and stored successfully in SQLite!")
        messagebox.showinfo("Success", "Data fetched and stored successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to fetch data: {e}")
        log_message(f"Error: {e}")

def export_to_csv():

    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

    
        cursor.execute("SELECT * FROM stock_data")
        data = cursor.fetchall()
        conn.close()

        if not data:
            messagebox.showwarning("No Data", "No data found in the database to export.")
            log_message("No data found in the database to export.")
            return


        csv_file = "stock_data.csv"

        # Write to CSV (add frequency column to the header)
        with open(csv_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "Ticker", "Date", "Open", "High", "Low", "Close", "Volume", "Frequency"])
            writer.writerows(data)

        log_message(f"Data exported to {csv_file} successfully!")
        messagebox.showinfo("Success", f"Data exported to {csv_file} successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to export data: {e}")
        log_message(f"Error exporting data: {e}")


# DEEP LEARNING ANALYSIS CODE

def run_final_prediction():
    try:
        log_message("Loading dataset from 'stock_data.csv'...")
        data = pd.read_csv('stock_data.csv')
 
        data.rename(columns={
            "Date": "date",
            "Ticker": "ticker",
            "Close": "close",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Volume": "volume",
            "Frequency": "frequency"
        }, inplace=True, errors='ignore')

        log_message(f"Data Shape: {data.shape}")
        if data.shape[0] == 0:
            messagebox.showerror("No Data", "stock_data.csv is empty.")
            return

        
        log_message("Sample of the data:\n" + str(data.sample(min(5, len(data)))))

        
        data['date'] = pd.to_datetime(data['date'])
        log_message("Date column converted to datetime.")

        
        selected_ticker = ticker_var.get()
        if not selected_ticker:
            messagebox.showerror("Ticker Error", "Please select a ticker to analyze.")
            return

        ticker_data = data[data['ticker'] == selected_ticker].copy()
        log_message(f"{selected_ticker} total records: {len(ticker_data)}")

        if len(ticker_data) == 0:
            messagebox.showerror("No Data", f"No data found for ticker {selected_ticker} in stock_data.csv.")
            return

        
        log_message(f"Plotting Open/Close for {selected_ticker}...")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(ticker_data['date'], ticker_data['close'], c="r", label="Close", marker="+")
        ax1.plot(ticker_data['date'], ticker_data['open'], c="g", label="Open", marker="^")
        ax1.set_title(f"{selected_ticker} - Open vs. Close")
        ax1.legend()
        display_plot(fig1, f"{selected_ticker} Open/Close Analysis")

        
        log_message(f"Plotting Volume for {selected_ticker}...")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(ticker_data['date'], ticker_data['volume'], c='purple', marker='*')
        ax2.set_title(f"{selected_ticker} Volume")
        display_plot(fig2, f"{selected_ticker} Volume Analysis")

        # Prepare data for LSTM
        close_data = ticker_data[['close']]
        dataset = close_data.values
        if len(dataset) < 60:
            messagebox.showerror("Not Enough Data", 
                f"Only {len(dataset)} rows for {selected_ticker}. Need >= 60 rows for LSTM.")
            return

        
        training_size = int(np.ceil(len(dataset) * 0.75))
        log_message(f"Training Data Size: {training_size} (out of {len(dataset)})")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        train_data = scaled_data[:training_size]
        x_train, y_train = [], []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        log_message("Building LSTM model...")
        model = keras.Sequential([
                keras.layers.Input(shape=(x_train.shape[1], 1)),  # Input layer
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.LSTM(64),
                keras.layers.Dense(32),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(1)
            ])


        model.compile(optimizer='adam', loss='mean_squared_error')
        log_message("Training the LSTM model (epochs=10)...")
        model.fit(x_train, y_train, epochs=10, verbose=1)
        log_message("Model training completed!")

        # Testing / Predictions
        test_data = scaled_data[training_size - 60:, :]
        x_test, y_test = [], dataset[training_size:]

        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])

        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        mse = np.mean((predictions - y_test) ** 2)
        rmse = np.sqrt(mse)
        log_message(f"MSE: {mse:.4f}")
        log_message(f"RMSE: {rmse:.4f}")

        # Split the dataset for plotting
        train_df = ticker_data.iloc[:training_size].copy()
        test_df = ticker_data.iloc[training_size:].copy()
        test_df['Predictions'] = predictions

        # 1) Get user-specified to_date
        end_date_str = to_date_var.get()
        end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")

        # 2) The day after user-specified to_date
        predicted_next_date = end_date_dt + pd.Timedelta(days=1)

        # 3) Make a single-step prediction using the last 60 points
        last_60_data = scaled_data[-60:]  # last 60 points of the entire dataset
        X_next = np.reshape(last_60_data, (1, 60, 1))
        next_day_pred_scaled = model.predict(X_next)
        next_day_pred = scaler.inverse_transform(next_day_pred_scaled)

        # 4) Print next-day prediction in UI
        log_message(f"Predicted close price for the day after {end_date_dt.strftime('%Y-%m-%d')} "
                    f"({predicted_next_date.strftime('%Y-%m-%d')}): "
                    f"{next_day_pred[0,0]:.2f}")

        # Plot everything
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.plot(train_df['date'], train_df['close'], label="Train")
        ax3.plot(test_df['date'], test_df['close'], label="Actual")
        ax3.plot(test_df['date'], test_df['Predictions'], label="Predicted")

        # Single point for the day after 'to_date'
        ax3.scatter(predicted_next_date, next_day_pred[0, 0],
                    color='blue', marker='o', s=100, label="Next Day after to_date")

        ax3.set_title(f"{selected_ticker} Stock Close Price Prediction\n+ Next Day after {end_date_dt.strftime('%Y-%m-%d')}")
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Close')
        ax3.legend()

        display_plot(fig3, f"{selected_ticker} Prediction vs Actual + 1-Day after to_date")

        log_message("Deep learning analysis completed successfully!")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        log_message(f"Error in deep learning analysis: {e}")

# HELPER FUNCTIONS


def display_plot(fig, title_str="Figure"):
    
#    Display a Matplotlib figure inside a new Tkinter Toplevel window.

    top = tk.Toplevel(root)
    top.title(title_str)
    canvas = FigureCanvasTkAgg(fig, master=top)
    canvas.draw()
    canvas.get_tk_widget().pack()

def log_message(message):
    """
    Print messages in the Tkinter Text widget instead of the console.
    """
    output_text.insert(tk.END, message + "\n")
    output_text.see(tk.END)  # Auto-scroll to the bottom


# MAIN TKINTER APP

create_table()

root = tk.Tk()
root.title("Financial Data Fetcher & Deep Learning Analysis")
root.geometry("600x600")

# Frame for Inputs
input_frame = tk.Frame(root)
input_frame.pack(pady=10)

# From Date
tk.Label(input_frame, text="From Date (YYYY-MM-DD):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
from_date_var = tk.StringVar()
from_date_entry = DateEntry(input_frame, textvariable=from_date_var, width=12, background='darkblue',
                            foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
from_date_entry.grid(row=0, column=1, padx=5, pady=5)

# To Date
tk.Label(input_frame, text="To Date (YYYY-MM-DD):").grid(row=1, column=0, padx=5, pady=5, sticky="e")
to_date_var = tk.StringVar()
to_date_entry = DateEntry(input_frame, textvariable=to_date_var, width=12, background='darkblue',
                          foreground='white', borderwidth=2, date_pattern='yyyy-mm-dd')
to_date_entry.grid(row=1, column=1, padx=5, pady=5)

# Ticker Symbol
tk.Label(input_frame, text="Enter Company Name:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
ticker_var = tk.StringVar()
ticker_entry = tk.Entry(input_frame, textvariable=ticker_var, width=20)
ticker_entry.grid(row=2, column=1, padx=5, pady=5)

# Frequency
tk.Label(input_frame, text="Select Frequency:").grid(row=3, column=0, padx=5, pady=5, sticky="e")
freq_var = tk.StringVar()
freq_dropdown = ttk.Combobox(input_frame, textvariable=freq_var, values=["1d", "1wk", "1mo"], state="readonly")
freq_dropdown.grid(row=3, column=1, padx=5, pady=5)
freq_dropdown.current(0) 

# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

fetch_button = tk.Button(button_frame, text="Fetch Data", command=fetch_and_store_data)
fetch_button.grid(row=0, column=0, padx=10)

export_button = tk.Button(button_frame, text="Export to CSV", command=export_to_csv)
export_button.grid(row=0, column=1, padx=10)

analyze_button = tk.Button(button_frame, text="Analyze (final prediction)", command=run_final_prediction)
analyze_button.grid(row=0, column=2, padx=10)

# Text widget for logs/outputs
output_text = tk.Text(root, height=15, width=70, wrap="word")
output_text.pack(pady=10)

root.mainloop()