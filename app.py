import streamlit as st
import os
import gdown
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.utils import to_networkx

import sys
# Tambahkan direktori kerja saat ini ke dalam path Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ... baru kemudian import yang lain
from fraud_detection.datasets import EllipticDataset
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
st.markdown("Menggunakan **Graph Neural Networks (GNN)** untuk mendeteksi transaksi ilegal pada jaringan Bitcoin.")

# ==========================================
# FUNGSI DOWNLOAD DARI DRIVE
# ==========================================
def download_dataset_files():
    """Mengunduh file dataset dari Google Drive jika belum ada di lokal"""
    # Dictionary mapping: Nama File Lokal -> File ID Google Drive
    files_to_download = {
        "elliptic_txs_features.csv": "1PcpbycYW06JdIm2vfrYjHuVPR1ApNOgJ",
        "elliptic_txs_edgelist.csv": "1bARgKjr3BWVCigrIhrQ1SdgSrJ4JABBq",
        "elliptic_txs_classes.csv": "1V7pbbbo34hidCgn7xwDegqOp1qVST0tB"
    }
    
    # Buat folder data jika belum ada (sesuaikan dengan config kamu)
    data_dir = "data/elliptic_bitcoin_dataset"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for filename, file_id in files_to_download.items():
        output_path = os.path.join(data_dir, filename)
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading {filename} from Drive..."):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output_path, quiet=False)

# ==========================================
# SIDEBAR: CONFIG & MODEL SELECTION
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model")

model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("GAT (Graph Attention Network)", "GIN (Graph Isomorphism Network)", "GCN (Graph Convolutional Network)")
)

if "GAT" in model_type:
    config_file = "configs/elliptic_gat.yaml"
    model_name = "elliptic_gat"
    ModelClass = GAT
elif "GIN" in model_type:
    config_file = "configs/elliptic_gin.yaml"
    model_name = "elliptic_gin"
    ModelClass = GIN
else:
    config_file = "configs/elliptic_gcn.yaml"
    model_name = "elliptic_gcn"
    ModelClass = GCN

# ==========================================
# FUNGSI LOAD DATA & MODEL (CACHED)
# ==========================================

@st.cache_resource
def load_data_and_config(cfg_path):
    try:
        # 1. Pastikan file terdownload
        download_dataset_files()
        
        # 2. Load Config
        conf = OmegaConf.load(cfg_path)

        # 3. Inisialisasi Dataset (Pastikan path di .yaml mengarah ke data/elliptic_bitcoin_dataset/...)
        ds = EllipticDataset(conf)
        graph_data = ds.pyg_dataset()

        node_ids = ds.features_df.index.values
        id_map = {node_id: i for i, node_id in enumerate(node_ids)}

        return conf, graph_data, id_map, ds
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None, None, None, None

@st.cache_resource
def load_trained_model(model_name, _ModelClass, _config, input_dim):
    try:
        path = f"weights/{model_name}.pt"
        if not os.path.exists(path):
            return None

        checkpoint = torch.load(path, map_location="cpu")
        _config.model.input_dim = input_dim
        model = _ModelClass(_config.model).double()
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================

with st.spinner('Memproses dataset graf...'):
    config, graph, id_map, raw_dataset = load_data_and_config(config_file)

if graph is not None:
    st.sidebar.success("✅ Dataset & Graph Loaded!")
    
    model = load_trained_model(model_name, ModelClass, config, graph.num_node_features)

    if model is None:
        st.error(f"❌ File Model `weights/{model_name}.pt` tidak ditemukan!")
    else:
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 Cek Transaksi")
            example_illicit = raw_dataset.illicit_ids[0] if len(raw_dataset.illicit_ids) > 0 else "2304203"
            tx_id_input = st.text_input("Masukkan Transaction ID (Node ID):", value=str(example_illicit))
            analyze_btn = st.button("Analisa Transaksi", type="primary")

        with col2:
            if analyze_btn:
                try:
                    tx_id = int(tx_id_input)
                    if tx_id in id_map:
                        node_idx = id_map[tx_id]
                        
                        with torch.no_grad():
                            logits = model(graph)
                            probs = torch.sigmoid(logits)
                            pred_prob = probs[node_idx].item()
                            prediction = 1 if pred_prob > 0.5 else 0
                        
                        st.subheader("Hasil Analisis")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Transaction ID", tx_id)
                        m2.metric("Fraud Prob", f"{pred_prob:.4f}")
                        
                        if prediction == 1:
                            m3.error("FRAUD!")
                        else:
                            m3.success("SAFE")

                        # Visualisasi (Tetap menggunakan kode Anda sebelumnya)
                        st.divider()
                        st.subheader("🕸️ Visualisasi Koneksi")
                        edge_index_np = graph.edge_index.numpy()
                        neighbors = edge_index_np[1, edge_index_np[0] == node_idx]
                        
                        G_vis = nx.Graph()
                        G_vis.add_node(tx_id, color='red' if prediction == 1 else 'green')
                        for n in neighbors:
                            # Map back to original ID if possible
                            G_vis.add_edge(tx_id, n)
                        
                        fig, ax = plt.subplots()
                        nx.draw(G_vis, with_labels=True, ax=ax)
                        st.pyplot(fig)
                    else:
                        st.warning("ID tidak ditemukan.")
                except Exception as e:
                    st.error(f"Error: {e}")

