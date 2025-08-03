# clean_history_csv.py
#
# Deskripsi:
# Skrip ini membaca file 'history.csv' yang diekspor dari MT5,
# membersihkan nama kolom yang tidak standar, memperbaiki format angka,
# dan menyimpannya sebagai file baru yang rapi bernama 'history_cleaned.csv'.

import pandas as pd
import logging
import re

# --- Konfigurasi ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = 'history.csv'
OUTPUT_FILE = 'history_cleaned.csv'

def clean_numeric_value(value):
    """Membersihkan string angka dari berbagai format secara robust."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            # Hapus semua spasi
            s = str(value).strip().replace(' ', '')
            # Hitung jumlah titik
            dots = s.count('.')
            # Jika ada lebih dari satu titik, anggap sebagai pemisah ribuan
            # dan hapus semua kecuali yang terakhir.
            if dots > 1:
                s = s.replace('.', '', dots - 1)
            return float(s)
        except (ValueError, TypeError):
            return None # Kembalikan None jika konversi gagal
    return None

def main():
    """Fungsi utama untuk membersihkan file CSV."""
    try:
        df = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
        logging.info(f"Berhasil membaca {len(df)} baris dari '{INPUT_FILE}'.")
    except FileNotFoundError:
        logging.error(f"File '{INPUT_FILE}' tidak ditemukan. Pastikan file ada di direktori yang sama.")
        return
    except Exception as e:
        logging.error(f"Gagal membaca file CSV: {e}")
        return

    # Menghapus kolom terakhir jika kosong (sering terjadi saat ekspor dari MT5)
    if df.columns[-1].startswith('Unnamed'):
        df = df.iloc[:, :-1]

    original_columns = df.columns.tolist()
    
    # Membersihkan nama kolom: huruf kecil, ganti spasi dengan _, hapus karakter aneh
    df.columns = [re.sub(r'\.\d+$', '', col).strip().lower().replace(' ', '_').replace('/', '') for col in original_columns]
    
    # Menangani kolom duplikat ('Time', 'Price') dari ekspor MT5
    cols = df.columns.tolist()
    time_indices = [i for i, name in enumerate(cols) if name == 'time']
    price_indices = [i for i, name in enumerate(cols) if name == 'price']

    if len(time_indices) > 1:
        cols[time_indices[0]] = 'open_time'
        cols[time_indices[1]] = 'close_time'
    if len(price_indices) > 1:
        cols[price_indices[0]] = 'open_price'
        cols[price_indices[1]] = 'close_price'
    df.columns = cols
    
    logging.info(f"Nama kolom asli: {original_columns}")
    logging.info(f"Nama kolom distandarkan: {df.columns.tolist()}")

    # Mengganti nama 'position' menjadi 'ticket' untuk konsistensi
    if 'position' in df.columns:
        df.rename(columns={'position': 'ticket'}, inplace=True)
        logging.info("Mengganti nama kolom 'position' menjadi 'ticket'.")

    # Membersihkan kolom-kolom numerik
    numeric_cols = ['open_price', 'close_price', 'profit', 'volume', 'commission', 'swap', 's_l', 't_p']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric_value)

    # Menghapus baris di mana data penting gagal dibersihkan
    df.dropna(subset=['open_price', 'profit', 'ticket'], inplace=True)
    df['ticket'] = df['ticket'].astype(int)
    logging.info(f"Jumlah baris setelah pembersihan data numerik: {len(df)}.")

    # Memilih dan menyusun ulang kolom-kolom penting
    final_cols_order = [
        'ticket', 'open_time', 'symbol', 'type', 'volume', 'open_price', 's_l', 't_p', 
        'close_time', 'close_price', 'commission', 'swap', 'profit'
    ]
    # Filter hanya kolom yang ada di dataframe
    existing_cols = [col for col in final_cols_order if col in df.columns]
    cleaned_df = df[existing_cols]

    try:
        cleaned_df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"Berhasil! {len(cleaned_df)} baris data yang bersih telah disimpan ke '{OUTPUT_FILE}'.")
    except Exception as e:
        logging.error(f"Gagal menyimpan file yang sudah dibersihkan: {e}")

if __name__ == '__main__':
    main()
