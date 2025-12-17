import sys
import os

# FIX: Tambahkan path agar folder 'fraud_detection' dikenali sebagai modul oleh Streamlit
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import gdown
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.utils import to_networkx

# Import modul lokal
from fraud_detection.datasets import EllipticDataset
from fraud_detection.models import GAT, GCN, GIN

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Bitcoin Fraud Detection GNN",
    page_icon="🔍",
    layout="wide"
)

st.title("🕵️ Bitcoin Fraud Detection System")
st.markdown("Sistem deteksi pencucian uang Bitcoin berbasis **Graph Neural Networks**.")

# ==========================================
# FUNGSI DOWNLOAD OTOMATIS (DATASET & RUNS)
# ==========================================

def download_from_drive():
    """Mengunduh semua file yang diperlukan untuk deployment"""
    # 1. Dataset Files
    data_dir = "data/elliptic_bitcoin_dataset"
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_files = {
        "elliptic_txs_features.csv": "1PcpbycYW06JdIm2vfrYjHuVPR1ApNOgJ",
        "elliptic_txs_edgelist.csv": "1bARgKjr3BWVCigrIhrQ1SdgSrJ4JABBq",
        "elliptic_txs_classes.csv": "1V7pbbbo34hidCgn7xwDegqOp1qVST0tB"
    }
    
    # 2. Training Runs (Weights & Logs)
    runs_base = "runs"
    os.makedirs(runs_base, exist_ok=True)
    
    training_folders = {
        "Dec01_11-14-07_LAPTOP-52M1BM4J": "1z4k4H5zZNh5thrXWp-QKUjWI32TBEOWV",
        "Dec01_11-36-51_LAPTOP-52M1BM4J": "1lIYyT3AJ_ah2AovWuO10M1xxKk06OPqF",
        "Dec01_11-39-34_LAPTOP-52M1BM4J": "1mF_XI9KWQ0Qd7iRj2j44cHNtEhk6i3hc",
        "Dec17_00-11-02_DESKTOP-P5BLEAQ": "1H5JKGuQQVLvI8IW2tPIyurm4aV7ddAJI",
        "Nov30_01-44-53_LAPTOP-52M1BM4J": "1MxXLDIkxBdYVYMDKr0pwKMMosd4kb1BK"
    }

    # Proses Download Dataset
    for name, fid in dataset_files.items():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            with st.spinner(f"Downloading {name}..."):
                gdown.download(id=fid, output=path, quiet=True)

    # Proses Download Folders Runs
    for folder_name, fid in training_folders.items():
        path = os.path.join(runs_base, folder_name)
        if not os.path.exists(path):
            with st.spinner(f"Downloading Training Results: {folder_name}..."):
                url = f"https://drive.google.com/drive/folders/{fid}"
                gdown.download_folder(url, output=path, quiet=True, use_cookies=False)

# ==========================================
# SIDEBAR & LOAD LOGIC
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model")

model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("GAT (Graph Attention Network)", "GIN (Graph Isomorphism Network)", "GCN (Graph Convolutional Network)")
)

if "GAT" in model_type:
    config_file, model_name, ModelClass = "configs/elliptic_gat.yaml", "elliptic_gat", GAT
elif "GIN" in model_type:
    config_file, model_name, ModelClass = "configs/elliptic_gin.yaml", "elliptic_gin", GIN
else:
    config_file, model_name, ModelClass = "configs/elliptic_gcn.yaml", "elliptic_gcn", GCN

@st.cache_resource
def load_all_assets(cfg_path):
    try:
        # Jalankan download sebelum load
        download_from_drive()
        
        # Load Config
        conf = OmegaConf.load(cfg_path)
        
        # Override paths agar sesuai dengan lokasi download
        conf.dataset.features_path = "data/elliptic_bitcoin_dataset/elliptic_txs_features.csv"
        conf.dataset.edges_path = "data/elliptic_bitcoin_dataset/elliptic_txs_edgelist.csv"
        conf.dataset.classes_path = "data/elliptic_bitcoin_dataset/elliptic_txs_classes.csv"

        ds = EllipticDataset(conf)
        graph_data = ds.pyg_dataset()
        id_map = {node_id: i for i, node_id in enumerate(ds.features_df.index.values)}
        
        return conf, graph_data, id_map, ds
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None

@st.cache_resource
def find_and_load_model(model_name, _ModelClass, _config, input_dim):
    try:
        import glob
        # Cari file .pt di folder weights atau folder runs secara rekursif
        pt_files = glob.glob(f"runs/**/{model_name}.pt", recursive=True) or \
                   glob.glob(f"weights/{model_name}.pt") or \
                   glob.glob(f"runs/**/*.pt", recursive=True)

        if not pt_files:
            return None

        path = pt_files[0]
        checkpoint = torch.load(path, map_location="cpu")
        _config.model.input_dim = input_dim
        model = _ModelClass(_config.model).double()
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

# ==========================================
# MAIN APP
# ==========================================

with st.spinner('Menyiapkan data dan model (ini mungkin memakan waktu beberapa menit pada jalankan pertama)...'):
    config, graph, id_map, raw_dataset = load_all_assets(config_file)

if graph is not None:
    st.sidebar.success("✅ Dataset Ready!")
    model = find_and_load_model(model_name, ModelClass, config, graph.num_node_features)

    if model is None:
        st.error(f"❌ File Model untuk {model_type} tidak ditemukan di folder 'runs'.")
    else:
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 Analisis Transaksi")
            example = raw_dataset.illicit_ids[0] if len(raw_dataset.illicit_ids) > 0 else "2304203"
            tx_id_input = st.text_input("Transaction ID:", value=str(example))
            
            if st.button("Jalankan Prediksi GNN", type="primary"):
                try:
                    tx_id = int(tx_id_input)
                    if tx_id in id_map:
                        node_idx = id_map[tx_id]
                        with torch.no_grad():
                            logits = model(graph)
                            prob = torch.sigmoid(logits)[node_idx].item()
                        
                        st.metric("Probability of Fraud", f"{prob*100:.2f}%")
                        if prob > 0.5:
                            st.error("🚨 TERDETEKSI ILLICIT")
                        else:
                            st.success("✅ TRANSAKSI AMAN")
                    else:
                        st.warning("ID tidak ditemukan dalam graf.")
                except:
                    st.error("Gagal memproses ID.")

        with col2:
            st.info("Sistem ini memproses data graf secara real-time untuk melihat relasi antar dompet.")
            # Visualisasi bisa ditambahkan di sini sesuai kode sebelumnya
