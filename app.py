import sys
import os
import glob

# --- 1. PATH FIX (Wajib di Streamlit Cloud) ---
# Menambahkan direktori root ke sys.path agar folder 'fraud_detection' terdeteksi
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import streamlit as st
import gdown
import torch
from omegaconf import OmegaConf

# --- 2. DEFINE PLACEHOLDERS & IMPORT ---
# Inisialisasi variabel agar tidak menyebabkan NameError jika import gagal
GAT, GCN, GIN = None, None, None

try:
    from fraud_detection.datasets import EllipticDataset
    from fraud_detection.models import GAT, GCN, GIN
except ImportError as e:
    st.error(f"Kritis: Gagal mengimpor modul internal. Detail: {e}")
    st.stop() # Hentikan eksekusi jika modul dasar tidak ditemukan

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Bitcoin Fraud Detection", layout="wide")

# ==========================================
# DOWNLOAD LOGIC (Tanpa Dataset Lokal)
# ==========================================
def download_assets():
    """Mengunduh weights dan data dari Drive"""
    # Buat folder jika belum ada
    os.makedirs("data/elliptic_bitcoin_dataset", exist_ok=True)
    os.makedirs("runs", exist_ok=True)

    # File dataset
    dataset_files = {
        "elliptic_txs_features.csv": "1PcpbycYW06JdIm2vfrYjHuVPR1ApNOgJ",
        "elliptic_txs_edgelist.csv": "1bARgKjr3BWVCigrIhrQ1SdgSrJ4JABBq",
        "elliptic_txs_classes.csv": "1V7pbbbo34hidCgn7xwDegqOp1qVST0tB"
    }

    for name, fid in dataset_files.items():
        path = f"data/elliptic_bitcoin_dataset/{name}"
        if not os.path.exists(path):
            gdown.download(id=fid, output=path, quiet=True)

    # Folder Runs (Weights)
    training_runs = {
        "run1": "1z4k4H5zZNh5thrXWp-QKUjWI32TBEOWV",
        "run2": "1lIYyT3AJ_ah2AovWuO10M1xxKk06OPqF"
    }
    for folder, fid in training_runs.items():
        path = f"runs/{folder}"
        if not os.path.exists(path):
            gdown.download_folder(url=f"https://drive.google.com/drive/folders/{fid}", 
                                  output=path, quiet=True, use_cookies=False)

# ==========================================
# MAIN INTERFACE
# ==========================================
st.sidebar.header("⚙️ Model Selection")
model_type = st.sidebar.selectbox("Pilih Arsitektur:", ("GAT", "GIN", "GCN"))

# Mapping Config (Menggunakan variabel yang sudah diimpor)
model_map = {
    "GAT": ("configs/elliptic_gat.yaml", "elliptic_gat", GAT),
    "GIN": ("configs/elliptic_gin.yaml", "elliptic_gin", GIN),
    "GCN": ("configs/elliptic_gcn.yaml", "elliptic_gcn", GCN)
}

config_path, model_name, ModelClass = model_map[model_type]

# Cek apakah kelas model tersedia
if ModelClass is None:
    st.error(f"Kelas model untuk {model_type} tidak tersedia. Periksa file models.py.")
    st.stop()

@st.cache_resource
def initialize_app(cfg_p):
    download_assets()
    conf = OmegaConf.load(cfg_p)
    # Sesuaikan path config secara runtime
    conf.dataset.features_path = "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
    conf.dataset.edges_path = "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
    conf.dataset.classes_path = "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
    
    ds = EllipticDataset(conf)
    graph_data = ds.pyg_dataset()
    id_map = {node_id: i for i, node_id in enumerate(ds.features_df.index.values)}
    return conf, graph_data, id_map

# Load Data
with st.spinner("Mengunduh dan menyiapkan data..."):
    config, graph, node_to_idx = initialize_app(config_path)

# Sisanya adalah logika prediksi...
st.success("Aplikasi Berhasil Dimuat!")
