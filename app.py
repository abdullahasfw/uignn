import streamlit as st
import os
import gdown
import torch
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric  # <--- SAYA TAMBAHKAN INI AGAR TIDAK ERROR
from torch_geometric.utils import to_networkx

# Import modules dari project kamu
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
st.markdown("Menggunakan **Graph Neural Networks (GNN)** untuk mendeteksi transaksi ilegal (Money Laundering) pada jaringan Bitcoin.")

# ==========================================
# SIDEBAR: CONFIG & MODEL SELECTION
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model")

# PERHATIKAN: Default pilih GAT karena file GCN belum ada di folder weights kamu
model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("GAT (Graph Attention Network)", "GIN (Graph Isomorphism Network)", "GCN (Graph Convolutional Network)")
)

# Mapping pilihan ke nama file config/weight
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


def download_features_if_needed(output_path):
    FILE_ID = "1PcpbycYW06JdIm2vfrYjHuVPR1ApNOgJ"
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, output_path, quiet=False)

@st.cache_resource
def load_data_and_config(cfg_path):
    try:
        conf = OmegaConf.load(cfg_path)

        # === DOWNLOAD FEATURES DARI GOOGLE DRIVE ===
        download_features_if_needed(conf.dataset.features_path)
        # ==========================================

        conf.features_path = conf.dataset.features_path
        conf.edges_path = conf.dataset.edges_path
        conf.classes = conf.dataset.classes

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
    """Load file .pt (Weights)"""
    try:
        path = f"weights/{model_name}.pt"
        # Cek manual apakah file ada sebelum load
        import os
        if not os.path.exists(path):
            return None # Return None jika file tidak ada

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

# 1. Load Data
with st.spinner('Sedang memuat data graf yang besar (harap tunggu sebentar)...'):
    config, graph, id_map, raw_dataset = load_data_and_config(config_file)

if graph is not None:
    st.sidebar.success("✅ Dataset & Graph Loaded!")
    
    # 2. Load Model
    model = load_trained_model(model_name, ModelClass, config, graph.num_node_features)

    # JIKA MODEL TIDAK DITEMUKAN (GCN kasusnya di sini)
    if model is None:
        st.error(f"❌ File Model `weights/{model_name}.pt` tidak ditemukan!")
        st.warning("⚠️ **Penyebab:** Model jenis ini belum pernah ditraining sebelumnya.")
        st.info("💡 **Solusi:** Silakan pilih model **GAT** atau **GIN** di Sidebar sebelah kiri, karena file weights-nya sudah tersedia di folder project kamu.")
    
    else:
        # Jika model ada, tampilkan interface
        st.divider()
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔍 Cek Transaksi")
            example_illicit = raw_dataset.illicit_ids[0] if len(raw_dataset.illicit_ids) > 0 else 0
            st.markdown(f"**Contoh ID Illicit (Fraud):** `{example_illicit}`")
            
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
                        m2.metric("Fraud Probability", f"{pred_prob:.4f}")
                        
                        if prediction == 1:
                            m3.error("FRAUD DETECTED!")
                            st.error(f"🚨 Transaksi ini terdeteksi sebagai **ILLICIT (Ilegal)** dengan keyakinan {(pred_prob*100):.2f}%")
                        else:
                            m3.success("SAFE")
                            st.success(f"✅ Transaksi ini terdeteksi sebagai **LICIT (Legal)**.")

                        st.divider()
                        st.subheader("🕸️ Visualisasi Koneksi (Neighbors)")
                        with st.spinner("Menggambar koneksi transaksi..."):
                            # Perbaikan pemanggilan torch_geometric
                            subset_nodes, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                                node_idx, num_hops=1, edge_index=graph.edge_index, relabel_nodes=True
                            )
                            
                            edge_index_np = graph.edge_index.numpy()
                            neighbors = edge_index_np[1, edge_index_np[0] == node_idx]
                            
                            G_vis = nx.Graph()
                            G_vis.add_node(tx_id, color='red' if prediction == 1 else 'green', size=500)
                            
                            for neighbor_idx in neighbors:
                                G_vis.add_node(neighbor_idx, color='gray', size=200)
                                G_vis.add_edge(tx_id, neighbor_idx)
                            
                            fig, ax = plt.subplots(figsize=(8, 5))
                            pos = nx.spring_layout(G_vis)
                            colors = [nx.get_node_attributes(G_vis, 'color').get(node, 'gray') for node in G_vis.nodes()]
                            sizes = [nx.get_node_attributes(G_vis, 'size').get(node, 200) for node in G_vis.nodes()]
                            
                            nx.draw(G_vis, pos, ax=ax, node_color=colors, node_size=sizes, with_labels=False)
                            st.pyplot(fig)
                    else:
                        st.warning("Transaction ID tidak ditemukan.")
                except ValueError:
                    st.error("Masukkan ID berupa angka.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {e}")