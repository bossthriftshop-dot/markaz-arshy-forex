# data_fetching.py
#
# Deskripsi:
# Versi ini telah diperbarui dengan koneksi MT5 yang lebih tangguh.
# Menambahkan batas waktu (timeout) dan logging yang lebih detail
# untuk mencegah skrip macet dan mendiagnosis masalah koneksi.

from __future__ import annotations

import pandas as pd
import MetaTrader5 as mt5
import time
import threading
import logging
from functools import lru_cache
from typing import Optional, List, Dict, Tuple

# --- Utility: Timeframe mapping & validator ---
def validate_timeframe(tf: str):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    if tf not in tf_map:
        raise ValueError(f"Invalid timeframe: {tf}")
    return tf_map[tf]

def validate_symbol(symbol: str):
    if not isinstance(symbol, str) or len(symbol) < 3:
        raise ValueError(f"Invalid symbol: {symbol}")

def validate_bars(bars: int):
    if not isinstance(bars, int) or bars <= 0:
        raise ValueError(f"Bars must be positive integer, got: {bars}")

# ====================================================================
# --- PERBAIKAN: Koneksi MT5 yang Lebih Tangguh dengan Timeout ---
# ====================================================================
def robust_mt5_init(mt5_path: str, retry: int = 3, sleep_sec: float = 2.0) -> bool:
    """
    Mencoba menginisialisasi koneksi ke MT5 dengan retry dan timeout.
    """
    for attempt in range(1, retry + 1):
        logging.info(f"[MT5 Connect] Upaya {attempt}/{retry}: Mencoba inisialisasi koneksi...")
        # Menambahkan timeout 10 detik (10000 milidetik)
        initialized = mt5.initialize(path=mt5_path, timeout=10000)
        
        if initialized:
            logging.info("[MT5 Connect] Koneksi berhasil.")
            return True
            
        logging.warning(f"[MT5 Connect] Upaya {attempt}/{retry}: Inisialisasi GAGAL. Mencoba lagi dalam {sleep_sec} detik...")
        time.sleep(sleep_sec)
        
    logging.error("[MT5 Connect] Semua upaya untuk menginisialisasi koneksi MT5 gagal.")
    return False
# ====================================================================

# --- Main Data Fetching Function ---
def get_candlestick_data(symbol: str, tf: str, bars: int, mt5_path: str, retry: int = 2) -> Optional[pd.DataFrame]:
    """
    Mengambil data candlestick dari MetaTrader 5, tangguh dan otomatis retry.
    """
    try:
        validate_symbol(symbol)
        tf_mt5 = validate_timeframe(tf)
        validate_bars(bars)
    except Exception as ve:
        logging.error(f"Parameter validation error: {ve}")
        return None

    for attempt in range(1, retry+1):
        if not robust_mt5_init(mt5_path, retry=1):
            time.sleep(1) # Beri jeda jika inisialisasi gagal
            continue
        try:
            rates = mt5.copy_rates_from_pos(symbol, tf_mt5, 0, bars)
            if rates is None or len(rates) == 0:
                logging.warning(f"Tidak ada data dari MT5 untuk {symbol} di {tf}. Upaya {attempt}")
                mt5.shutdown()
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            mt5.shutdown()
            return df
        except Exception as e:
            logging.error(f"Error di get_candlestick_data [{symbol}/{tf}]: {e}", exc_info=True)
            mt5.shutdown()
            time.sleep(1)
    logging.error(f"Gagal mengambil data untuk {symbol} di {tf} setelah beberapa kali percobaan.")
    return None

# --- Bulk Fetcher ---
def get_bulk_candlestick_data(
    symbols: List[str],
    tfs: List[str],
    bars: int,
    mt5_path: str
) -> Dict[Tuple[str, str], Optional[pd.DataFrame]]:
    """
    Ambil data untuk banyak symbol + TF sekaligus.
    """
    result = {}
    for symbol in symbols:
        for tf in tfs:
            df = get_candlestick_data(symbol, tf, bars, mt5_path)
            result[(symbol, tf)] = df
    return result

# --- Memory Cache Class ---
class DataCache:
    def __init__(self, expiry_seconds: int = 300):
        self._cache = {}
        self._lock = threading.Lock()
        self.expiry_seconds = expiry_seconds

    @lru_cache(maxsize=100)
    def get_candlestick_data(self, symbol: str, timeframe: str, bars: int, mt5_path: str) -> Optional[pd.DataFrame]:
        cache_key = f"{symbol}_{timeframe}_{bars}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached['timestamp']) < self.expiry_seconds:
                return cached['data']
            data = get_candlestick_data(symbol, timeframe, bars, mt5_path)
            self._cache[cache_key] = {
                'data': data,
                'timestamp': now
            }
            return data
