import sys
import os

# --- FIX: Tambahkan root direktori ke sys.path agar modul 'fraud_detection' terbaca ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import streamlit as st
import gdown
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import glob
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.utils import to_networkx

# Import lokal (Pastikan folder fraud_detection memiliki file __init__.py)
try:
    from fraud_detection.datasets import EllipticDataset
    from fraud_detection.models import GAT, GCN, GIN
except ImportError as e:
    st.error(f"Gagal mengimport modul: {e}")
    st.info("Pastikan folder 'fraud_detection' memiliki file '__init__.py' kosong.")

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Bitcoin Fraud GNN", page_icon="🔍", layout="wide")
st.title("🕵️ Bitcoin Fraud Detection System")

# ==========================================
# FUNGSI DOWNLOAD (DATASET & MODEL)
# ==========================================
def download_assets():
    """Mengunduh Dataset dan Hasil Training dari Drive"""
    # 1. Dataset
    data_dir = "data/elliptic_bitcoin_dataset"
    os.makedirs(data_dir, exist_ok=True)
    dataset_files = {
        "elliptic_txs_features.csv": "1PcpbycYW06JdIm2vfrYjHuVPR1ApNOgJ",
        "elliptic_txs_edgelist.csv": "1bARgKjr3BWVCigrIhrQ1SdgSrJ4JABBq",
        "elliptic_txs_classes.csv": "1V7pbbbo34hidCgn7xwDegqOp1qVST0tB"
    }
    for name, fid in dataset_files.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            gdown.download(id=fid, output=path, quiet=True)

    # 2. Folder Runs (Hasil Training)
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)
    training_runs = {
        "Dec01_11-14": "1z4k4H5zZNh5thrXWp-QKUjWI32TBEOWV",
        "Dec01_11-36": "1lIYyT3AJ_ah2AovWuO10M1xxKk06OPqF",
        "Dec01_11-39": "1mF_XI9KWQ0Qd7iRj2j44cHNtEhk6i3hc",
        "Dec17_00-11": "1H5JKGuQQVLvI8IW2tPIyurm4aV7ddAJI",
        "Nov30_01-44": "1MxXLDIkxBdYVYMDKr0pwKMMosd4kb1BK"
    }
    for folder_name, fid in training_runs.items():
        path = os.path.join(runs_dir, folder_name)
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh Weights {folder_name}..."):
                gdown.download_folder(url=f"https://drive.google.com/drive/folders/{fid}", 
                                      output=path, quiet=True, use_cookies=False)

# ==========================================
# LOGIK MODEL & DATA
# ==========================================
st.sidebar.header("⚙️ Konfigurasi")
model_type = st.sidebar.selectbox("Arsitektur Model:", ("GAT", "GIN", "GCN"))

# Mapping Config
model_map = {
    "GAT": ("configs/elliptic_gat.yaml", "elliptic_gat", GAT),
    "GIN": ("configs/elliptic_gin.yaml", "elliptic_gin", GIN),
    "GCN": ("configs/elliptic_gcn.yaml", "elliptic_gcn", GCN)
}
config_file, model_file_name, ModelClass = model_map[model_type]

@st.cache_resource
def load_all_data(cfg_path):
    download_assets()
    conf = OmegaConf.load(cfg_path)
    # Paksa path ke direktori download
    conf.dataset.features_path = "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
    conf.dataset.edges_path = "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
    conf.dataset.classes_path = "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"
    
    ds = EllipticDataset(conf)
    graph_data = ds.pyg_dataset()
    id_map = {node_id: i for i, node_id in enumerate(ds.features_df.index.values)}
    return conf, graph_data, id_map, ds

@st.cache_resource
def load_model_weights(name, _Class, _conf, in_dim):
    # Cari file .pt di folder runs
    files = glob.glob(f"runs/**/{name}.pt", recursive=True) + glob.glob("runs/**/*.pt", recursive=True)
    if not files: return None
    
    ckpt = torch.load(files[0], map_location="cpu")
    _conf.model.input_dim = in_dim
    model = _Class(_conf.model).double()
    model.load_state_dict(ckpt)
    model.eval()
    return model

# --- Main Execution ---
with st.spinner("Memuat sistem (Hanya lama saat pertama kali)..."):
    config, graph, id_map, dataset = load_all_data(config_file)

if graph:
    model = load_model_weights(model_file_name, ModelClass, config, graph.num_node_features)
    
    if model:
        st.success(f"✅ Model {model_type} Siap!")
        tx_id = st.text_input("Transaction ID:", value="2304203")
        if st.button("Analisa"):
            if int(tx_id) in id_map:
                idx = id_map[int(tx_id)]
                with torch.no_grad():
                    prob = torch.sigmoid(model(graph))[idx].item()
                st.metric("Fraud Probability", f"{prob*100:.2f}%")
                if prob > 0.5: st.error("🚨 TERDETEKSI FRAUD")
                else: st.success("✅ TRANSAKSI LICIT")
            else:
                st.warning("ID tidak ditemukan.")
    else:
        st.error("Weights tidak ditemukan di folder runs.")
