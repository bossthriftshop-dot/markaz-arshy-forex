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
SYMBOL = "BTCUSD"
TIMEFRAMES_LIST = ["M5", "M15", "H1"]
DATA_CACHE = DataCache()
MODEL_DIR = "gng_models"
MT5_TERMINAL_PATH = r"C:\\Program Files\\ExclusiveMarkets MetaTrader5\\terminal64.exe"
MAX_POSITION_PER_TF = 5
API_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
SERVER_URL = "http://127.0.0.1:5000/api/internal/submit_signal"
SECRET_KEY = "c1b086d4-a681-48df-957f-6fcc35a82f6d"
MIN_DISTANCE_PIPS_PER_TF = { "M5": 100, "M15": 150, "M30": 200, "H1": 250 }
active_signals: Dict[str, Dict[str, any]] = {}
SIGNAL_COOLDOWN_MINUTES = 1
SIGNAL_MEMORY_MINUTES = 1

# --- Parameter Strategi ---
CONFIDENCE_THRESHOLD = 1.0
XGBOOST_CONFIDENCE_THRESHOLD = 0.65

# --- Inisialisasi Model AI ---
signal_cooldown: Dict[str, datetime] = {}
XGBOOST_MODEL_PATH = f"xgboost_model_{SYMBOL}.json"
xgb_model = None
try:
    logging.info(f"Mencoba memuat model AI untuk {SYMBOL} dari '{XGBOOST_MODEL_PATH}'...")
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(XGBOOST_MODEL_PATH)
    logging.info(f"Model AI untuk {SYMBOL} berhasil dimuat.")
except Exception as e:
    logging.error(f"GAGAL memuat model AI untuk {SYMBOL}: {e}. Bot akan berjalan TANPA konfirmasi AI.")
    xgb_model = None

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
    gng_models, gng_feature_stats = initialize_gng_models(
        symbol=SYMBOL, timeframes=TIMEFRAMES_LIST, model_dir=MODEL_DIR,
        mt5_path=MT5_TERMINAL_PATH, get_data_func=get_candlestick_data
    )
    auto_learners = {tf: AutoLearningModule(SYMBOL, tf) for tf in TIMEFRAMES_LIST}
    adaptive_gng = {tf: AdaptiveGNGLearning(gng_models.get(tf)) for tf in TIMEFRAMES_LIST if gng_models.get(tf)}
    regime_detector = MarketRegimeDetector()

    logging.info("="*50)
    logging.info("Bot Trading AI v2.2 (Mode Scalping) Siap Beraksi!")
    logging.info(f"Threshold Aturan: {CONFIDENCE_THRESHOLD} | Threshold AI: {XGBOOST_CONFIDENCE_THRESHOLD:.0%}")
    logging.info("="*50)

    try:
        while True:
            try:
                # --- Logika Cooldown ---
                cooldown_key = SYMBOL
                if cooldown_key in signal_cooldown:
                    time_since_signal = (datetime.now() - signal_cooldown[cooldown_key]).total_seconds() / 60
                    if time_since_signal < SIGNAL_COOLDOWN_MINUTES:
                        logging.info(f"Sabar dulu... Masih dalam masa tenang untuk {cooldown_key}. Sisa: {SIGNAL_COOLDOWN_MINUTES - time_since_signal:.1f} menit.")
                        time.sleep(20)
                        continue
                    else:
                        logging.info(f"Masa tenang untuk {cooldown_key} selesai.")
                        del signal_cooldown[cooldown_key]

                logging.info("-" * 50)
                logging.info("Memulai siklus analisis baru...")

                # --- Loop Analisis per Timeframe ---
                for tf in TIMEFRAMES_LIST:
                    logging.info(f"Menganalisis timeframe {tf}...")
                    opp = analyze_tf_opportunity(
                        symbol=SYMBOL, tf=tf, mt5_path=MT5_TERMINAL_PATH,
                        gng_model=gng_models.get(tf), gng_feature_stats=gng_feature_stats,
                        confidence_threshold=0.0, # Ambil semua potensi peluang, keputusan di loop ini
                        min_distance_pips_per_tf=MIN_DISTANCE_PIPS_PER_TF, htf_bias=None
                    )

                    if opp and opp.get('score') is not None and opp.get('signal') != "WAIT":
                        if opp['score'] >= CONFIDENCE_THRESHOLD:
                            logging.info(f"✅ SINYAL ENTRY! Peluang {opp['signal']} di {tf} memenuhi syarat. Skor: {opp['score']:.2f} (Min: {CONFIDENCE_THRESHOLD}).")

                            # --- Validasi AI ---
                            if xgb_model and opp.get('features') is not None and opp['features'].size > 0:
                                features = np.array(opp['features']).reshape(1, -1)
                                win_probability = xgb_model.predict_proba(features)[0][1]
                                logging.info(f"Meminta pendapat AI... Prediksi: {win_probability:.2%} kemungkinan WIN.")
                                if win_probability < XGBOOST_CONFIDENCE_THRESHOLD:
                                    logging.warning(f"AI menyarankan tidak mengambil ini. Keyakinan ({win_probability:.2%}) di bawah standar ({XGBOOST_CONFIDENCE_THRESHOLD:.0%}). Peluang dilewati.")
                                    continue
                                logging.info(f"Lampu hijau dari AI! Melanjutkan ke pemeriksaan keamanan.")

                            # --- Pemeriksaan Keamanan ---
                            open_pos_count = get_open_positions_per_tf(SYMBOL, tf, MT5_TERMINAL_PATH)
                            logging.info(f"Pemeriksaan 1/3: Batas posisi. Terbuka: {open_pos_count} (Maks: {MAX_POSITION_PER_TF}).")
                            if open_pos_count >= MAX_POSITION_PER_TF:
                                logging.warning(f"   -> Gagal. Batas posisi terbuka tercapai.")
                                continue

                            logging.info("   -> Lolos.")
                            mt5_orders = get_active_orders(SYMBOL, MT5_TERMINAL_PATH)
                            recent_signals = get_recent_signals_from_memory(active_signals, minutes=SIGNAL_MEMORY_MINUTES)
                            all_known_orders = list(dict.fromkeys(mt5_orders + recent_signals))
                            entry_price = float(opp['entry_price_chosen'])
                            point_value = 0.1
                            logging.info(f"Pemeriksaan 2/3: Jarak entry. Entry di {entry_price} vs order lain: {all_known_orders}.")
                            if not is_far_enough(entry_price, all_known_orders, point_value, MIN_DISTANCE_PIPS_PER_TF.get(tf, 1100)):
                                logging.warning("   -> Gagal. Terlalu dekat dengan order lain.")
                                continue

                            logging.info("   -> Lolos.")

                            # --- Persiapan & Pengiriman Sinyal ---
                            order_type_to_use = opp.get('order_type')
                            if not order_type_to_use:
                                logging.warning(f"Tipe order tidak ditemukan di sinyal, menggunakan default market order: '{opp['signal']}'")
                                order_type_to_use = opp['signal']

                            signal_json = build_signal_format(
                                symbol=SYMBOL, entry_price=entry_price, direction=opp['signal'],
                                sl=float(opp['sl']), tp=float(opp['tp']), order_type=order_type_to_use
                            )
                            sig_id = make_signal_id(signal_json)

                            logging.info(f"Pemeriksaan 3/3: Duplikasi. Sinyal ID: {sig_id}. Memori sinyal aktif: {list(active_signals.keys())}")
                            if sig_id in active_signals:
                                logging.warning("   -> Gagal. Sinyal ini duplikat dari yang baru saja dikirim.")
                                continue

                            logging.info("   -> Lolos. Siap dikirim!")
                            send_status = send_signal_to_server(SYMBOL, signal_json, API_KEY, SERVER_URL, SECRET_KEY)
                            if send_status == 'SUCCESS' or send_status == 'REJECTED':
                                if send_status == 'SUCCESS':
                                    active_signals[sig_id] = {'signal_json': signal_json, 'tf': tf, 'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'info': opp['info'], 'status': 'pending'}
                                    logging.info("Sinyal berhasil dikirim!")
                                else:
                                    logging.warning("Server menolak sinyal (kemungkinan duplikat).")
                                signal_cooldown[SYMBOL] = datetime.now()
                                logging.info(f"Cooldown {SIGNAL_COOLDOWN_MINUTES} menit diaktifkan untuk {SYMBOL}.")
                                break # Hentikan loop timeframe karena sinyal berhasil dikirim/ditangani
                            else:
                                logging.error(f"Pengiriman GAGAL karena error koneksi.")

                        else:
                            logging.info(f"⏳ SINYAL WAIT. Peluang di {tf} terdeteksi, namun skor ({opp['score']:.2f}) di bawah threshold ({CONFIDENCE_THRESHOLD}).")

                logging.info(f"Siklus selesai. Istirahat 20 detik...")
                time.sleep(20)
            except Exception as e:
                logging.critical(f"Terjadi error kritis di loop utama: {e}", exc_info=True)
                time.sleep(20)
    except KeyboardInterrupt:
        logging.info("Perintah berhenti diterima. Menyimpan riwayat...")
        for learner in auto_learners.values():
            learner.save_history()
        logging.info("Semua data berhasil disimpan. Sampai jumpa!")

if __name__ == '__main__':
    main()