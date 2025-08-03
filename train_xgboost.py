# train_xgboost.py
#
# Deskripsi:
# Versi final. Skrip ini sekarang menggunakan parameter hyper-tuned yang optimal
# untuk melatih model AI yang paling akurat dan kuat untuk setiap simbol.

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================================================================
# --- PERUBAHAN: Menyimpan Parameter Optimal dari Hasil Tuning ---
# ====================================================================
# Parameter ini didapat dari menjalankan 'tune_hyperparameters.py'.
# Jika Anda menjalankan tuning lagi dan mendapatkan hasil yang lebih baik,
# Anda bisa memperbarui nilai-nilai di bawah ini.
BEST_PARAMS = {
    "BTCUSD": {
    "colsample_bytree": 0.749816047538945,
    "learning_rate": 0.20014286128198325,
    "max_depth": 4,
    "n_estimators": 171,
    "subsample": 0.8394633936788146
},
    "XAUUSD": {
    "colsample_bytree": 0.9754210836063001,
    "learning_rate": 0.010155753168202867,
    "max_depth": 2,
    "n_estimators": 157,
    "subsample": 0.8099025726528951

    },
    # Default parameter jika simbol baru muncul
    "DEFAULT": {
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "max_depth": 4,
        "n_estimators": 150,
        "subsample": 0.8
    }
}
# ====================================================================


def train_model_for_symbol(symbol_data, symbol_name):
    """
    Fungsi untuk melatih model menggunakan parameter yang sudah dioptimalkan.
    """
    model_output_path = f"xgboost_model_{symbol_name}.json"
    logging.info(f"--- Memulai Pelatihan FINAL untuk Simbol: {symbol_name} ---")

    # --- Persiapan Data ---
    features = [record['gng_input_features_on_signal'] for record in symbol_data]
    labels = [1 if record['result'] == "WIN" else 0 for record in symbol_data]
    X = np.array(features)
    y = np.array(labels)

    if len(X) < 50:
        logging.warning(f"Data latih untuk {symbol_name} tidak mencukupi. Pelatihan dilewati.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    logging.info(f"Data {symbol_name} dibagi menjadi {len(X_train)} data latih dan {len(X_test)} data uji.")

    # --- Pelatihan Model dengan Parameter Optimal ---
    # Mengambil parameter terbaik untuk simbol ini, atau default jika tidak ada.
    params = BEST_PARAMS.get(symbol_name, BEST_PARAMS["DEFAULT"])
    logging.info(f"Menggunakan parameter optimal untuk {symbol_name}: {params}")

    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        **params  # Menggunakan parameter terbaik yang sudah kita simpan
    )
    
    xgb_classifier.fit(X_train, y_train)
    logging.info("Pelatihan model final selesai.")

    # --- Evaluasi Kinerja ---
    y_pred_test = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, zero_division=0)
    logging.info(f"--- Hasil Evaluasi Final untuk {symbol_name} ---")
    logging.info(f"Akurasi   : {accuracy:.2%}")
    logging.info(f"Presisi   : {precision:.2%}")
    logging.info("-------------------------------------------")

    # --- Menyimpan Model ---
    xgb_classifier.save_model(model_output_path)
    logging.info(f"Model AI final untuk {symbol_name} telah disimpan ke '{model_output_path}'.")

def main():
    """
    Fungsi utama untuk memuat data dan mengorkestrasi pelatihan per simbol.
    """
    feedback_file_path = "trade_feedback_generated.json"
    try:
        with open(feedback_file_path, "r") as f:
            trade_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Gagal memuat '{feedback_file_path}': {e}.")
        return

    logging.info(f"Berhasil memuat {len(trade_data)} total record data latih.")
    df = pd.DataFrame(trade_data)
    df['clean_symbol'] = df['symbol'].apply(lambda s: re.sub(r'[cm]$', '', s).upper())
    
    grouped = df.groupby('clean_symbol')
    logging.info(f"Data akan dilatih untuk simbol: {list(grouped.groups.keys())}")

    for symbol_name, group_df in grouped:
        symbol_records = group_df.to_dict('records')
        train_model_for_symbol(symbol_records, symbol_name)

if __name__ == '__main__':
    main()
