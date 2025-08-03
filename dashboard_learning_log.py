import streamlit as st
import pandas as pd
import sqlite3
import numpy as np

st.set_page_config(page_title="Learning Log Dashboard", layout="wide", page_icon=":bar_chart:")

# --- LOAD DATA ---
conn = sqlite3.connect("learning_log.db")
df = pd.read_sql_query("SELECT * FROM learning_log", conn)
conn.close()

st.title("📊 Learning Analytics Dashboard")
st.write("Monitor performa, winrate, dan hasil learning AI Trading kamu secara real-time.")

# --- FILTER SYMBOL/TF ---
symbols = df['symbol'].unique().tolist()
timeframes = df['tf'].unique().tolist()
col1, col2 = st.columns(2)
with col1:
    selected_symbol = st.selectbox("Symbol", symbols, index=0)
with col2:
    selected_tf = st.selectbox("Timeframe", timeframes, index=0)
df_filtered = df[(df['symbol'] == selected_symbol) & (df['tf'] == selected_tf)]

# --- KPI ---
total_trades = len(df_filtered)
winrate = 100 * (df_filtered['pnl'] > 0).sum() / total_trades if total_trades else 0
total_pnl = df_filtered['pnl'].sum()
profit_curve = np.cumsum(df_filtered['pnl'].values) if total_trades else []
max_drawdown = min(profit_curve) if len(profit_curve) else 0

colA, colB, colC, colD = st.columns(4)
colA.metric("Total Trades", total_trades)
colB.metric("Winrate (%)", f"{winrate:.2f}", delta=None)
colC.metric("Total Profit (PnL)", f"{total_pnl:.2f}")
colD.metric("Max Drawdown", f"{max_drawdown:.2f}")

# --- PNL CHART ---
st.subheader("Cumulative PnL (Profit Curve)")
st.line_chart(profit_curve)

# --- TABLE DATA ---
st.subheader("Detail Trade Log")
st.dataframe(df_filtered, use_container_width=True)
