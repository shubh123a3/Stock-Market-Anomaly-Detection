import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam


# Function to load and preprocess data
@st.cache_data
def load_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Returns'] = data['Close'].pct_change()
    data['Volatility'] = data['Returns'].rolling(window=20).std()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    data['Upper_BB'] = data['MA20'] + (data['Close'].rolling(window=20).std() * 2)
    data['Lower_BB'] = data['MA20'] - (data['Close'].rolling(window=20).std() * 2)
    data = data.dropna()
    return data


# Function for EDA plots
def create_eda_plots(data):
    # Closing Price with MA20 and Bollinger Bands
    fig_close = go.Figure()
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig_close.add_trace(
        go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='20-day MA', line=dict(color='orange')))
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Upper_BB'], mode='lines', name='Upper BB',
                                   line=dict(color='gray', dash='dash')))
    fig_close.add_trace(go.Scatter(x=data.index, y=data['Lower_BB'], mode='lines', name='Lower BB',
                                   line=dict(color='gray', dash='dash')))
    fig_close.update_layout(title='Closing Price with 20-day MA and Bollinger Bands', xaxis_title='Date',
                            yaxis_title='Price')

    # Volume
    fig_volume = go.Figure()
    fig_volume.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume'))
    fig_volume.update_layout(title='Trading Volume Over Time', xaxis_title='Date', yaxis_title='Volume')

    # Returns
    fig_returns = go.Figure()
    fig_returns.add_trace(go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns'))
    fig_returns.update_layout(title='Daily Returns Over Time', xaxis_title='Date', yaxis_title='Returns')

    # Volatility
    fig_volatility = go.Figure()
    fig_volatility.add_trace(go.Scatter(x=data.index, y=data['Volatility'], mode='lines', name='Volatility'))
    fig_volatility.update_layout(title='20-Day Volatility Over Time', xaxis_title='Date', yaxis_title='Volatility')

    return fig_close, fig_volume, fig_returns, fig_volatility


# Anomaly detection functions (as previously defined)
def detect_zscore_anomalies(data, threshold=3):
    z_scores = np.abs((data['Close'] - data['Close'].mean()) / data['Close'].std())
    return z_scores > threshold


def detect_iforest_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    return iso_forest.fit_predict(X) == -1


def detect_dbscan_anomalies(data):
    scaler = StandardScaler()
    X = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    return dbscan.fit_predict(X) == -1


def detect_lstm_anomalies(data, sequence_length=20, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])

    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 4)),
        Dense(4)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    predictions = model.predict(X)
    mse = np.mean(np.power(y - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    anomalies = np.zeros(len(data))
    anomalies[sequence_length:] = mse > threshold
    return anomalies.astype(bool)


def detect_autoencoder_anomalies(data, threshold_percentile=95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'Returns', 'Volatility']])

    input_dim = scaled_data.shape[1]
    encoding_dim = 2

    input_layer = Input(shape=(input_dim,))
    encoder = Dense(8, activation="relu")(input_layer)
    encoder = Dense(4, activation="relu")(encoder)
    encoder = Dense(encoding_dim, activation="relu")(encoder)
    decoder = Dense(4, activation="relu")(encoder)
    decoder = Dense(8, activation="relu")(decoder)
    decoder = Dense(input_dim, activation="linear")(decoder)

    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    autoencoder.fit(scaled_data, scaled_data, epochs=100, batch_size=32, shuffle=True, verbose=0)

    predictions = autoencoder.predict(scaled_data)
    mse = np.mean(np.power(scaled_data - predictions, 2), axis=1)
    threshold = np.percentile(mse, threshold_percentile)

    return mse > threshold


# Streamlit app
st.title('Stock Price Anomaly Detection')

# Sidebar inputs
ticker = st.sidebar.text_input('Enter stock ticker', 'GME')
start_date = st.sidebar.date_input('Start date', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End date', pd.to_datetime('2023-12-31'))

# Load data
data = load_data(ticker, start_date, end_date)

# EDA Section
st.header('Exploratory Data Analysis')
fig_close, fig_volume, fig_returns, fig_volatility = create_eda_plots(data)

st.subheader('Closing Price with 20-day MA and Bollinger Bands')
st.plotly_chart(fig_close)

st.subheader('Trading Volume Over Time')
st.plotly_chart(fig_volume)

st.subheader('Daily Returns Over Time')
st.plotly_chart(fig_returns)

st.subheader('20-Day Volatility Over Time')
st.plotly_chart(fig_volatility)

# Basic Statistics
st.subheader('Basic Statistics')
st.write(data.describe())

# Correlation Matrix
st.subheader('Correlation Matrix')
corr_matrix = data[['Close', 'Volume', 'Returns', 'Volatility']].corr()
fig_corr = go.Figure(
    data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.index, y=corr_matrix.columns, colorscale='Viridis'))
fig_corr.update_layout(title='Correlation Matrix')
st.plotly_chart(fig_corr)

# Anomaly Detection Section
st.header('Anomaly Detection')

# Detect anomalies
zscore_anomalies = detect_zscore_anomalies(data)
iforest_anomalies = detect_iforest_anomalies(data)
dbscan_anomalies = detect_dbscan_anomalies(data)
lstm_anomalies = detect_lstm_anomalies(data)
autoencoder_anomalies = detect_autoencoder_anomalies(data)


# Create plots
def create_anomaly_plot(data, anomalies, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=data.index[anomalies], y=data['Close'][anomalies], mode='markers', name='Anomalies',
                             marker=dict(color='red', size=8)))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price')
    return fig


st.subheader('Z-Score Anomalies')
st.plotly_chart(create_anomaly_plot(data, zscore_anomalies, 'Z-Score Anomalies'))

st.subheader('Isolation Forest Anomalies')
st.plotly_chart(create_anomaly_plot(data, iforest_anomalies, 'Isolation Forest Anomalies'))

st.subheader('DBSCAN Anomalies')
st.plotly_chart(create_anomaly_plot(data, dbscan_anomalies, 'DBSCAN Anomalies'))

st.subheader('LSTM Anomalies')
st.plotly_chart(create_anomaly_plot(data, lstm_anomalies, 'LSTM Anomalies'))

st.subheader('Autoencoder Anomalies')
st.plotly_chart(create_anomaly_plot(data, autoencoder_anomalies, 'Autoencoder Anomalies'))

# Combined plot
st.subheader('All Models Comparison')
fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
fig.add_trace(
    go.Scatter(x=data.index[zscore_anomalies], y=data['Close'][zscore_anomalies], mode='markers', name='Z-Score',
               marker=dict(color='red', size=8, symbol='circle')))
fig.add_trace(go.Scatter(x=data.index[iforest_anomalies], y=data['Close'][iforest_anomalies], mode='markers',
                         name='Isolation Forest', marker=dict(color='green', size=8, symbol='square')))
fig.add_trace(
    go.Scatter(x=data.index[dbscan_anomalies], y=data['Close'][dbscan_anomalies], mode='markers', name='DBSCAN',
               marker=dict(color='blue', size=8, symbol='diamond')))
fig.add_trace(go.Scatter(x=data.index[lstm_anomalies], y=data['Close'][lstm_anomalies], mode='markers', name='LSTM',
                         marker=dict(color='purple', size=8, symbol='cross')))
fig.add_trace(go.Scatter(x=data.index[autoencoder_anomalies], y=data['Close'][autoencoder_anomalies], mode='markers',
                         name='Autoencoder', marker=dict(color='orange', size=8, symbol='star')))
fig.update_layout(title='All Models Comparison', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Summary statistics
st.subheader('Summary Statistics')
summary = pd.DataFrame({
    'Model': ['Z-Score', 'Isolation Forest', 'DBSCAN', 'LSTM', 'Autoencoder'],
    'Anomalies Detected': [
        zscore_anomalies.sum(),
        iforest_anomalies.sum(),
        dbscan_anomalies.sum(),
        lstm_anomalies.sum(),
        autoencoder_anomalies.sum()
    ]
})
st.table(summary)