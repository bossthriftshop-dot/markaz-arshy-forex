# main.py
#
# Deskripsi:
# Versi ini telah disesuaikan untuk strategi SCALPING yang lebih agresif.
# Logika entri dan tunggu (wait) kini dikontrol secara eksplisit di loop utama
# berdasarkan skor peluang dibandingkan dengan CONFIDENCE_THRESHOLD.

import logging
import os
import time
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta
from typing import Dict, List
import json

from data_fetching import get_candlestick_data, DataCache
from gng_model import initialize_gng_models
from signal_generator import (
    analyze_tf_opportunity,
    build_signal_format,
    make_signal_id,
    get_open_positions_per_tf,
    get_active_orders,
    is_far_enough
)
from learning import (
    AutoLearningModule,
    AdaptiveGNGLearning,
    MarketRegimeDetector,
    get_active_trades_results,
    apply_learning_adjustments
)
from server_comm import send_signal_to_server, cancel_signal

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TIMEFRAMES_LIST = ["M5", "M15", "H1"]
SYMBOL_CONFIGS = {
    "BTCUSD": {
        "min_distance_pips": {"M5": 100, "M15": 150, "H1": 250, "M30": 200},
        "point_value": 0.1
    },
    "XAUUSD": {
        "min_distance_pips": {"M5": 100, "M15": 150, "H1": 250, "M30": 200},
        "point_value": 0.01
    },
    "EURUSD": {
        "min_distance_pips": {"M5": 10, "M15": 15, "H1": 25, "M30": 20},
        "point_value": 0.0001
    },
    "USDJPY": {
        "min_distance_pips": {"M5": 10, "M15": 15, "H1": 25, "M30": 20},
        "point_value": 0.01
    }
}
SYMBOLS_TO_ANALYZE = ["BTCUSD", "EURUSD"] # <-- Daftar simbol yang akan dianalisis

DATA_CACHE = DataCache()
MODEL_DIR = "gng_models"
MT5_TERMINAL_PATH = r"C:\\Program Files\\ExclusiveMarkets MetaTrader5\\terminal64.exe"
MAX_POSITION_PER_TF = 5
API_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
SERVER_URL = "http://127.0.0.1:5000/api/internal/submit_signal"
SECRET_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
active_signals: Dict[str, Dict[str, any]] = {}
SIGNAL_COOLDOWN_MINUTES = 1
SIGNAL_MEMORY_MINUTES = 1

# --- Parameter Strategi ---
CONFIDENCE_THRESHOLD = 1.0
XGBOOST_CONFIDENCE_THRESHOLD = 0.65

# --- Inisialisasi Model AI ---
signal_cooldown: Dict[str, datetime] = {}
xgb_models: Dict[str, xgb.XGBClassifier] = {}
for symbol in SYMBOLS_TO_ANALYZE:
    model_path = f"xgboost_model_{symbol}.json"
    try:
        logging.info(f"Mencoba memuat model AI untuk {symbol} dari '{model_path}'...")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        xgb_models[symbol] = model
        logging.info(f"Model AI untuk {symbol} berhasil dimuat.")
    except Exception as e:
        logging.error(f"GAGAL memuat model AI untuk {symbol}: {e}. Bot akan berjalan TANPA konfirmasi AI untuk simbol ini.")

def get_recent_signals_from_memory(active_signals: Dict[str, Dict[str, any]], minutes: int) -> List[float]:
    recent_prices = []
    now = datetime.now()
    signals_to_clear = []

    for sig_id, signal_data in list(active_signals.items()):
        try:
            signal_time = datetime.strptime(signal_data['timestamp'], "%Y-%m-%d %H:%M:%S")
            if (now - signal_time).total_seconds() > (minutes * 60):
                signals_to_clear.append(sig_id)
            else:
                original_signal = signal_data['signal_json']
                entry_val_str = (original_signal.get("BuyEntry") or original_signal.get("SellEntry") or
                                 original_signal.get("BuyStop") or original_signal.get("SellStop") or
                                 original_signal.get("BuyLimit") or original_signal.get("SellLimit"))
                if entry_val_str:
                    recent_prices.append(float(entry_val_str))
        except (ValueError, KeyError):
            signals_to_clear.append(sig_id)

    if signals_to_clear:
        logging.info(f"Membersihkan {len(signals_to_clear)} sinyal lama dari memori (lebih dari {minutes} menit).")
        for sig_id in signals_to_clear:
            if sig_id in active_signals:
                del active_signals[sig_id]

    return recent_prices

def main() -> None:
    # --- Inisialisasi Multi-Simbol ---
    all_gng_models = {}
    all_gng_feature_stats = {}
    all_auto_learners = {}
    all_adaptive_gng = {}

    logging.info("Menginisialisasi model untuk semua simbol yang dikonfigurasi...")
    for symbol in SYMBOLS_TO_ANALYZE:
        logging.info(f"--- Memproses simbol: {symbol} ---")
        gng_models, gng_feature_stats = initialize_gng_models(
            symbol=symbol, timeframes=TIMEFRAMES_LIST, model_dir=MODEL_DIR,
            mt5_path=MT5_TERMINAL_PATH, get_data_func=get_candlestick_data
        )
        all_gng_models[symbol] = gng_models
        all_gng_feature_stats[symbol] = gng_feature_stats
        all_auto_learners[symbol] = {tf: AutoLearningModule(symbol, tf) for tf in TIMEFRAMES_LIST}
        all_adaptive_gng[symbol] = {tf: AdaptiveGNGLearning(gng_models.get(tf)) for tf in TIMEFRAMES_LIST if gng_models.get(tf)}

    regime_detector = MarketRegimeDetector()
    logging.info("="*50)
    logging.info("Bot Trading AI v2.3 (Mode Multi-Simbol) Siap Beraksi!")
    logging.info(f"Simbol dianalisis: {', '.join(SYMBOLS_TO_ANALYZE)}")
    logging.info(f"Threshold Aturan: {CONFIDENCE_THRESHOLD} | Threshold AI: {XGBOOST_CONFIDENCE_THRESHOLD:.0%}")
    logging.info("="*50)

    try:
        while True:
            # --- Loop Analisis per Simbol ---
            for symbol in SYMBOLS_TO_ANALYZE:
                try:
                    logging.info(f"--- Siklus Analisis untuk [{symbol}] ---")
                    symbol_config = SYMBOL_CONFIGS.get(symbol, {})
                    if not symbol_config:
                        logging.warning(f"Konfigurasi untuk simbol {symbol} tidak ditemukan. Dilewati.")
                        continue

                    # --- Logika Cooldown per Simbol ---
                    cooldown_key = symbol
                    if cooldown_key in signal_cooldown:
                        time_since_signal = (datetime.now() - signal_cooldown[cooldown_key]).total_seconds() / 60
                        if time_since_signal < SIGNAL_COOLDOWN_MINUTES:
                            logging.info(f"Sabar dulu... Masih dalam masa tenang untuk {cooldown_key}. Sisa: {SIGNAL_COOLDOWN_MINUTES - time_since_signal:.1f} menit.")
                            continue # Lanjut ke simbol berikutnya
                        else:
                            logging.info(f"Masa tenang untuk {cooldown_key} selesai.")
                            del signal_cooldown[cooldown_key]

                    # --- Loop Analisis per Timeframe ---
                    for tf in TIMEFRAMES_LIST:
                        logging.info(f"Menganalisis {symbol} pada timeframe {tf}...")

                        # Ambil komponen spesifik untuk simbol dan timeframe ini
                        gng_model_tf = all_gng_models.get(symbol, {}).get(tf)
                        gng_stats_tf = all_gng_feature_stats.get(symbol, {})
                        xgb_model_symbol = xgb_models.get(symbol)
                        min_dist_pips_tf = symbol_config.get("min_distance_pips", {})
                        point_value = symbol_config.get("point_value", 0.01)

                        opp = analyze_tf_opportunity(
                            symbol=symbol, tf=tf, mt5_path=MT5_TERMINAL_PATH,
                            gng_model=gng_model_tf, gng_feature_stats=gng_stats_tf,
                            confidence_threshold=0.0,
                            min_distance_pips_per_tf=min_dist_pips_tf, htf_bias=None
                        )

                        if opp and opp.get('score') is not None and opp.get('signal') != "WAIT":
                            if opp['score'] >= CONFIDENCE_THRESHOLD:
                                logging.info(f"✅ SINYAL ENTRY! Peluang {opp['signal']} di {symbol}-{tf} memenuhi syarat. Skor: {opp['score']:.2f}")

                                # --- Validasi AI ---
                                if xgb_model_symbol and opp.get('features') is not None and opp['features'].size > 0:
                                    features = np.array(opp['features']).reshape(1, -1)
                                    win_probability = xgb_model_symbol.predict_proba(features)[0][1]
                                    logging.info(f"Meminta pendapat AI... Prediksi: {win_probability:.2%} kemungkinan WIN.")
                                    if win_probability < XGBOOST_CONFIDENCE_THRESHOLD:
                                        logging.warning(f"AI menyarankan tidak mengambil ini. Keyakinan ({win_probability:.2%}) di bawah standar.")
                                        continue
                                    logging.info(f"Lampu hijau dari AI! Melanjutkan ke pemeriksaan keamanan.")

                                # --- Pemeriksaan Keamanan ---
                                open_pos_count = get_open_positions_per_tf(symbol, tf, MT5_TERMINAL_PATH)
                                if open_pos_count >= MAX_POSITION_PER_TF:
                                    logging.warning(f"Gagal. Batas posisi terbuka ({open_pos_count}/{MAX_POSITION_PER_TF}) tercapai untuk {tf}.")
                                    continue

                                mt5_orders = get_active_orders(symbol, MT5_TERMINAL_PATH)
                                recent_signals = get_recent_signals_from_memory(active_signals, minutes=SIGNAL_MEMORY_MINUTES)
                                all_known_orders = list(dict.fromkeys(mt5_orders + recent_signals))
                                entry_price = float(opp['entry_price_chosen'])

                                if not is_far_enough(entry_price, all_known_orders, point_value, min_dist_pips_tf.get(tf, 100)):
                                    logging.warning(f"Gagal. Terlalu dekat dengan order lain.")
                                    continue

                                # --- Persiapan & Pengiriman Sinyal ---
                                signal_json = build_signal_format(
                                    symbol=symbol, entry_price=entry_price, direction=opp['signal'],
                                    sl=float(opp['sl']), tp=float(opp['tp']), order_type=opp.get('order_type', opp['signal'])
                                )
                                sig_id = make_signal_id(signal_json)

                                if sig_id in active_signals:
                                    logging.warning("Gagal. Sinyal ini duplikat dari yang baru saja dikirim.")
                                    continue

                                logging.info("Lolos semua pemeriksaan. Siap dikirim!")
                                send_status = send_signal_to_server(symbol, signal_json, API_KEY, SERVER_URL, SECRET_KEY, tf)
                                if send_status in ['SUCCESS', 'REJECTED']:
                                    if send_status == 'SUCCESS':
                                        active_signals[sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'info': opp['info'], 'status': 'pending'}
                                        logging.info("Sinyal berhasil dikirim!")
                                    else:
                                        logging.warning("Server menolak sinyal (kemungkinan duplikat).")
                                    signal_cooldown[symbol] = datetime.now()
                                    break # Hentikan loop timeframe, lanjut ke simbol berikutnya
                                else:
                                    logging.error(f"Pengiriman GAGAL karena error koneksi.")

                            else:
                                logging.info(f"⏳ SINYAL WAIT. Peluang di {symbol}-{tf} terdeteksi, skor ({opp['score']:.2f}) di bawah threshold.")

                except Exception as e:
                    logging.critical(f"Terjadi error kritis saat menganalisis simbol {symbol}: {e}", exc_info=True)

                logging.info(f"Siklus analisis untuk {symbol} selesai. Istirahat 2 detik sebelum ke simbol berikutnya...")
                time.sleep(2)

            logging.info("Semua simbol telah dianalisis. Istirahat 20 detik sebelum siklus baru...")
            time.sleep(20)

    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Menyimpan riwayat...")
        for symbol_learners in all_auto_learners.values():
            for learner in symbol_learners.values():
                learner.save_history()
        logging.info("Semua data berhasil disimpan. Sampai jumpa!")

if __name__ == '__main__':
    main()