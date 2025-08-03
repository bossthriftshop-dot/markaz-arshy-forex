"""
learning.py (ADVANCED, AI/ML Integrated)
========================================

Modul ini memuat kelas dan fungsi terkait mekanisme pembelajaran berkelanjutan.
Integrasi penuh pipeline technical_indicators refactor & GNG model baru.
Mendukung logging ke database, auto save/load trade history, multi-symbol & multi-timeframe,
fitur vektor AI-ready, dan label otomatis.

Copyright (c) 2024.
"""

from __future__ import annotations

import logging
import os
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

from technical_indicators import (
    detect_structure,
    detect_order_blocks_multi,
    detect_fvg_multi,
    detect_engulfing,
    detect_pinbar,
    get_daily_high_low,
    get_pivot_points,
    extract_features_full,
    generate_label_fvg,
)

from gng_model import (
    get_gng_input_features_full,
    GrowingNeuralGas,
    get_gng_context,
)

from trade_logger import log_trade_to_db

# ========== AUTO LEARNING MODULE ==========

class AutoLearningModule:
    """
    Modul pembelajaran otomatis, logging ke database,
    auto save/load history ke file JSON per-symbol dan timeframe,
    serta output vektor fitur AI-ready.
    """
    def __init__(self, symbol: str, timeframe: str, max_history: int = 1000, history_dir: str = "learning_history"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.trade_history: List[Dict[str, any]] = []
        self.max_history = max_history
        self.history_dir = os.path.join(history_dir, self.symbol)
        os.makedirs(self.history_dir, exist_ok=True)
        self.history_path = os.path.join(self.history_dir, f"{self.timeframe}.json")
        self.load_history()  # Auto-load saat inisialisasi

    def save_history(self):
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.trade_history, f, indent=2, default=str)
            logging.info(f"[Learning] Trade history {self.symbol}-{self.timeframe} berhasil disimpan ke {self.history_path}")
        except Exception as e:
            logging.error(f"[Learning] Gagal menyimpan history {self.symbol}-{self.timeframe}: {e}")

    def load_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.trade_history = json.load(f)
                logging.info(f"[Learning] Trade history {self.symbol}-{self.timeframe} dimuat dari {self.history_path}")
            except Exception as e:
                logging.warning(f"[Learning] Gagal memuat history {self.symbol}-{self.timeframe}: {e}")

    def analyze_trade_result(self, trade_data: Dict[str, any], autosave: bool = True) -> Optional[Dict[str, any]]:
        current_price = trade_data['current_price']
        entry_price = trade_data['entry_price']
        direction = trade_data['direction']
        pnl = (current_price - entry_price) if direction == "BUY" else (entry_price - current_price)
        features = trade_data.get('features')

        # --- log data ke DB (eksternal + file history) ---
        log_data = {
            'timestamp': trade_data.get('timestamp', datetime.now().isoformat()),
            'symbol': trade_data.get('symbol', self.symbol),
            'tf': self.timeframe,
            'entry': entry_price,
            'exit': current_price,
            'pnl': pnl,
            'direction': direction,
            'features': features if features is not None else '',
            'setup': trade_data.get('setup', ''),
            'confidence': trade_data.get('confidence', 0.0),
            'regime': trade_data.get('regime', ''),
        }
        try:
            log_trade_to_db(log_data)
        except Exception as e:
            logging.error(f"Error logging trade to DB: {e}")

        # --- simpan ke history lokal (dan file, jika autosave) ---
        self.trade_history.append({
            'timestamp': log_data['timestamp'],
            'entry': entry_price,
            'exit': current_price,
            'pnl': pnl,
            'direction': direction,
            'features': features,
            'setup': trade_data.get('setup', ''),
            'confidence': trade_data.get('confidence', 0.0),
            'regime': trade_data.get('regime', ''),
        })
        if len(self.trade_history) > self.max_history:
            self.trade_history.pop(0)
        if autosave:
            self.save_history()
        return self.calculate_adjustments()

    def calculate_adjustments(self) -> Optional[Dict[str, any]]:
        if len(self.trade_history) < 10:
            return None
        recent_trades = self.trade_history[-10:]
        win_rate = sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades)
        adjustments: Dict[str, any] = {
            'confidence_threshold': None,
            'weight_adjustments': {}
        }
        if win_rate < 0.4:
            adjustments['confidence_threshold'] = 1.1
        elif win_rate > 0.6:
            adjustments['confidence_threshold'] = 0.95
        return adjustments

# ========== ADAPTIVE GNG LEARNING ==========

class AdaptiveGNGLearning:
    """
    Kelas pembelajaran adaptif untuk Growing Neural Gas.
    Update node terdekat berdasarkan hasil trade (multi-dimensi).
    """
    def __init__(self, gng_model: GrowingNeuralGas):
        self.gng_model = gng_model

    def auto_adjust_node(self, trade_result: Dict[str, any], features: np.ndarray) -> None:
        profit = trade_result.get('profit', 0.0)
        confidence = trade_result.get('confidence', 1.0)
        nodes_w = np.array([node['w'] for node in self.gng_model.nodes])
        distances = np.array([np.linalg.norm(features - node_w) for node_w in nodes_w])
        nearest_idx = np.argmin(distances)
        node = self.gng_model.nodes[nearest_idx]
        if profit > 0:
            adaptation_rate = min(0.1, profit * confidence)
            node['w'] += adaptation_rate * (features - node['w'])
        else:
            adaptation_rate = min(0.05, abs(profit) * confidence)
            node['w'] -= adaptation_rate * (features - node['w'])

# ========== MARKET REGIME DETECTOR ==========

class MarketRegimeDetector:
    """
    Mendeteksi kondisi pasar (trending, ranging, consolidating, choppy) dari data.
    """
    def __init__(self, lookback_period: int = 100):
        self.lookback_period = lookback_period

    def detect_current_regime(self, df: Optional[pd.DataFrame]) -> str:
        if df is None or len(df) < 20:
            return 'UNKNOWN'
        df = df.copy()
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        volatility = df['std20'].iloc[-1]
        avg_volatility = df['std20'].mean()
        price_change = abs(df['close'].iloc[-1] - df['close'].iloc[0])
        price_range = df['high'].max() - df['low'].min()
        trend_strength = price_change / price_range if price_range != 0 else 0
        if volatility > avg_volatility * 1.5:
            if trend_strength > 0.6:
                return 'TRENDING'
            else:
                return 'CHOPPY'
        else:
            if trend_strength > 0.3:
                return 'RANGING'
            else:
                return 'CONSOLIDATING'

    def adapt_to_regime(self, regime: str, base_threshold: float, base_distance: float) -> Dict[str, float]:
        params = {
            'TRENDING': {
                'confidence_threshold': base_threshold * 0.9,
                'min_distance_pips': base_distance * 1.2
            },
            'RANGING': {
                'confidence_threshold': base_threshold * 1.1,
                'min_distance_pips': base_distance * 0.8
            },
            'CHOPPY': {
                'confidence_threshold': base_threshold * 1.2,
                'min_distance_pips': base_distance * 1.5
            },
            'CONSOLIDATING': {
                'confidence_threshold': base_threshold,
                'min_distance_pips': base_distance
            },
            'UNKNOWN': {
                'confidence_threshold': base_threshold,
                'min_distance_pips': base_distance
            }
        }
        return params.get(regime, params['UNKNOWN'])

# ========== AKTIF TRADE RESULT & LEARNING LOOP UTILITY ==========

def get_active_trades_results(symbol: str, timeframe: str, mt5_path: str, get_data_func) -> Optional[Dict[str, any]]:
    """
    Mengambil hasil trade aktif dari MT5, dan mengembalikan data siap untuk pembelajaran.
    Otomatis extract fitur vektor AI-ready dari candle terbaru.
    """
    if not mt5.initialize(path=mt5_path):
        logging.error("MT5 init gagal di get_active_trades_results()")
        return None
    try:
        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            return None
        pos = positions[0]
        entry_price = float(pos.price_open)
        current_price = float(pos.price_current)
        direction = "BUY" if pos.type == mt5.POSITION_TYPE_BUY else "SELL"
        df = get_data_func(symbol, timeframe, 100, mt5_path)
        if df is not None:
            # =========== EXTRACT FEATURE VECTOR (AI READY) ===========
            structure, _ = detect_structure(df)
            ob = detect_order_blocks_multi(df, structure_filter=structure)
            fvg = detect_fvg_multi(df)
            patterns = detect_engulfing(df) + detect_pinbar(df)
            boundary = get_daily_high_low(df)
            pivot = get_pivot_points(df)
            features = extract_features_full(df, structure, ob, fvg, patterns, boundary, pivot)
            result = {
                'entry_price': entry_price,
                'current_price': current_price,
                'direction': direction,
                'profit': pos.profit,
                'volume': pos.volume,
                'features': features.tolist() if isinstance(features, np.ndarray) else features,
                'setup': {
                    'structure': structure,
                    'has_ob': len(ob) > 0,
                    'has_fvg': len(fvg) > 0
                }
            }
            return result
        return None
    except Exception as e:
        logging.error(f"Error di get_active_trades_results: {e}")
        return None
    finally:
        mt5.shutdown()

def apply_learning_adjustments(gng_model, adjustments: Optional[Dict[str, any]]) -> None:
    """
    Terapkan penyesuaian pembelajaran ke model GNG.
    """
    if not adjustments or not gng_model:
        return
    try:
        if 'weight_adjustments' in adjustments:
            weight_changes = adjustments['weight_adjustments']
            for node_idx, weight_delta in weight_changes.items():
                if node_idx < len(gng_model.nodes):
                    gng_model.nodes[node_idx]['w'] += weight_delta
                    logging.debug(f"Applied weight adjustment to node {node_idx}")
    except Exception as e:
        logging.error(f"Error applying learning adjustments: {e}")

# ========== MULTI-SYMBOL / MULTI-TIMEFRAME UTILITY ==========

def create_learners_for_symbols_timeframes(symbols: List[str], timeframes: List[str], max_history: int = 1000) -> Dict[str, Dict[str, AutoLearningModule]]:
    learners = {}
    for symbol in symbols:
        learners[symbol] = {}
        for tf in timeframes:
            learners[symbol][tf] = AutoLearningModule(symbol, tf, max_history=max_history)
    return learners

# ========== LABEL OTOMATIS UNTUK ML/BACKTEST ==========

def generate_labels_from_trades(trade_history: List[Dict[str, any]], pip_thresh: float = 0.001, horizon: int = 10) -> List[int]:
    """
    Label otomatis untuk supervised learning.
    """
    labels = []
    for trade in trade_history:
        # Label 1 jika pnl >= pip_thresh dalam horizon X (fitur dummy)
        if 'pnl' in trade and trade['pnl'] >= pip_thresh:
            labels.append(1)
        else:
            labels.append(0)
    return labels

# ========== END OF MODULE ==========
