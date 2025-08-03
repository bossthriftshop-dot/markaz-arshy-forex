# analyze_model.py
#
# Deskripsi:
# Skrip ini memuat model AI (XGBoost) yang sudah dilatih dan menganalisisnya
# untuk mengetahui fitur-fitur pasar mana yang paling berpengaruh dalam
# membuat keputusan trading. Hasilnya akan disimpan sebagai gambar grafik.
#
# Cara Penggunaan:
# 1. Pastikan Anda sudah berhasil melatih model (misal: 'xgboost_model_BTCUSD.json').
# 2. Jalankan skrip ini dari terminal: python analyze_model.py

import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import logging
import glob

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Daftar nama fitur sesuai dengan urutan di 'extract_features_full'
# Ini sangat penting agar grafik bisa terbaca dengan benar.
FEATURE_NAMES = [
    'last_price',
    'atr_dynamic',
    'distance_to_ob',
    'strength_of_ob',
    'distance_to_fvg',
    'strength_of_fvg',
    'has_engulfing_pattern',
    'distance_to_daily_high',
    'distance_to_daily_low',
    'distance_from_r1',
    'distance_from_s1'
]

def analyze_model(model_path):
    """Memuat satu model dan mem-plot feature importance-nya."""
    try:
        # Ekstrak nama simbol dari path file
        symbol_name = model_path.split('_')[-1].replace('.json', '')
        logging.info(f"--- Menganalisis Model untuk: {symbol_name} ---")

        # Muat model
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(model_path)
        
        # Atur nama fitur di dalam model
        xgb_model.get_booster().feature_names = FEATURE_NAMES

        # Dapatkan skor kepentingan fitur
        importance_scores = xgb_model.get_booster().get_score(importance_type='weight')
        
        if not importance_scores:
            logging.warning(f"Tidak dapat mengambil skor kepentingan untuk model {symbol_name}.")
            return

        # Buat plot
        fig, ax = plt.subplots(figsize=(10, 8))
        xgb.plot_importance(importance_scores, ax=ax, height=0.8, title=f'Feature Importance untuk {symbol_name}',
                              xlabel='Skor Kepentingan (Weight)', ylabel='Fitur Pasar')
        
        # Simpan plot sebagai file gambar
        output_image_path = f'feature_importance_{symbol_name}.png'
        plt.tight_layout()
        plt.savefig(output_image_path)
        logging.info(f"Grafik untuk {symbol_name} telah disimpan ke '{output_image_path}'.")
        plt.close(fig) # Tutup plot agar tidak ditampilkan di layar

    except Exception as e:
        logging.error(f"Gagal menganalisis model dari '{model_path}': {e}")

def main():
    """Mencari semua file model yang ada dan menganalisisnya satu per satu."""
    # Cari semua file model xgboost di direktori saat ini
    model_files = glob.glob('xgboost_model_*.json')
    
    if not model_files:
        logging.error("Tidak ada file model ('xgboost_model_*.json') yang ditemukan.")
        logging.error("Pastikan Anda sudah menjalankan 'train_xgboost.py' terlebih dahulu.")
        return
        
    logging.info(f"Ditemukan {len(model_files)} model untuk dianalisis: {model_files}")
    
    for model_file in model_files:
        analyze_model(model_file)

if __name__ == '__main__':
    # Pastikan Anda sudah menginstal matplotlib
    # Jalankan dari terminal: pip install matplotlib
    main()
