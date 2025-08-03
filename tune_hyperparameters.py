# tune_hyperparameters.py
#
# Deskripsi:
# Versi ini disesuaikan untuk strategi SCALPING.
# Skrip ini melakukan 'Hyperparameter Tuning' menggunakan RandomizedSearchCV
# untuk menemukan kombinasi pengaturan terbaik bagi model XGBoost Anda secara
# lebih cepat dan efisien.

import json
import numpy as np
import pandas as pd
import xgboost as xgb
import logging
import re
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint, uniform

# --- Konfigurasi Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def tune_model_for_symbol(symbol_data, symbol_name):
    """
    Fungsi untuk melakukan tuning hyperparameter untuk satu simbol spesifik
    menggunakan RandomizedSearchCV yang lebih cepat.
    """
    logging.info(f"--- Memulai Tuning Hyperparameter (Scalping) untuk: {symbol_name} ---")

    # --- Persiapan Data ---
    features = [record['gng_input_features_on_signal'] for record in symbol_data]
    labels = [1 if record['result'] == "WIN" else 0 for record in symbol_data]

    X = np.array(features)
    y = np.array(labels)

    if len(X) < 100:
        logging.warning(f"Data untuk {symbol_name} ({len(X)} baris) mungkin tidak cukup untuk tuning yang efektif. Hasil mungkin kurang optimal.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- Mendefinisikan Distribusi Parameter untuk Diuji (Fokus Scalping) ---
    # Untuk scalping, kita cenderung memilih model yang lebih sederhana (max_depth rendah)
    # untuk menghindari overfitting pada noise pasar jangka pendek.
    param_dist = {
        'max_depth': randint(2, 5),  # Menguji kedalaman pohon yang lebih dangkal (2, 3, 4)
        'learning_rate': uniform(0.01, 0.2), # Mencari learning rate dalam rentang 0.01 - 0.21
        'n_estimators': randint(100, 250), # Jumlah pohon
        'subsample': uniform(0.6, 0.4), # Rentang 0.6 - 1.0
        'colsample_bytree': uniform(0.6, 0.4) # Rentang 0.6 - 1.0
    }

    # Inisialisasi model XGBoost
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss'
    )

    # Inisialisasi RandomizedSearchCV
    # n_iter=50 berarti akan menguji 50 kombinasi acak dari parameter di atas.
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=param_dist,
        n_iter=50, # Jumlah iterasi pencarian, bisa dinaikkan jika ingin lebih teliti
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    logging.info(f"Memulai RandomizedSearchCV untuk {symbol_name}. Ini akan menguji 50 kombinasi parameter...")
    random_search.fit(X_train, y_train)
    logging.info(f"Tuning untuk {symbol_name} selesai.")

    # --- Menampilkan Hasil Terbaik ---
    logging.info(f"--- Hasil Tuning Optimal (Scalping) untuk {symbol_name} ---")
    logging.info(f"Skor Akurasi Terbaik: {random_search.best_score_:.2%}")
    logging.info("Parameter Terbaik yang Ditemukan:")
    best_params_str = json.dumps(random_search.best_params_, indent=4)
    print(best_params_str)
    logging.info("-------------------------------------------------")
    logging.info(f"SALIN parameter di atas dan tempel ke dalam skrip 'train_xgboost.py' untuk model {symbol_name}.")


def main():
    """
    Fungsi utama untuk memuat data dan mengorkestrasi tuning per simbol.
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
    logging.info(f"Data akan di-tuning untuk simbol: {list(grouped.groups.keys())}")

    for symbol_name, group_df in grouped:
        symbol_records = group_df.to_dict('records')
        tune_model_for_symbol(symbol_records, symbol_name)

if __name__ == '__main__':
    main()
