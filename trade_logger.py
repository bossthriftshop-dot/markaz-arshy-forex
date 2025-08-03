import sqlite3
import json
from datetime import datetime

def log_trade_to_db(trade_data, db_path='learning_log.db'):
    """
    Logging hasil trade ke database SQLite.
    trade_data: dict dengan field minimal:
        - timestamp (str/datetime)
        - symbol (str)
        - tf (str)
        - entry (float)
        - exit (float)
        - pnl (float)
        - direction (str)
        - features (str/list/dict)
        - setup (str/dict)
        - confidence (float, optional)
        - regime (str, optional)
    """
    # Normalisasi timestamp
    ts = trade_data.get('timestamp')
    if ts is None:
        ts = datetime.now().isoformat()
    elif isinstance(ts, datetime):
        ts = ts.isoformat()

    # Konversi features dan setup ke JSON string supaya aman untuk semua tipe
    features_str = json.dumps(trade_data.get('features', ''))
    setup_str = json.dumps(trade_data.get('setup', ''))

    # Siapkan kolom-kolom wajib
    vals = (
        ts,
        trade_data.get('symbol', 'XAUUSD'),
        trade_data.get('tf', 'M15'),
        float(trade_data.get('entry', 0.0)),
        float(trade_data.get('exit', 0.0)),
        float(trade_data.get('pnl', 0.0)),
        trade_data.get('direction', ''),
        features_str,
        setup_str,
        float(trade_data.get('confidence', 0.0)),
        trade_data.get('regime', ''),
    )
    # Eksekusi insert
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS learning_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            tf TEXT,
            entry REAL,
            exit REAL,
            pnl REAL,
            direction TEXT,
            features TEXT,
            setup TEXT,
            confidence REAL,
            regime TEXT
        )
    """)
    c.execute("""
        INSERT INTO learning_log (
            timestamp, symbol, tf, entry, exit, pnl, direction, features, setup, confidence, regime
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, vals)
    conn.commit()
    conn.close()
