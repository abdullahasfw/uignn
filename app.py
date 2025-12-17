import streamlit as st
import torch
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.data import Data

# Import model definition
from fraud_detection.models import GAT, GCN, GIN

st.set_page_config(page_title="Bitcoin Fraud Detection", page_icon="🕵️", layout="wide")

st.title("🕵️ Bitcoin Fraud Detection System")
st.markdown("Menggunakan **Graph Neural Networks (GNN)**. Pilih nomor urut node untuk mendeteksi apakah transaksi tersebut Fraud (Illicit) atau Aman (Licit).")

# ==========================================
# 1. SETUP MODEL
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model")
model_type = st.sidebar.selectbox("Pilih Arsitektur Model:", ("GAT", "GIN", "GCN"))

@st.cache_resource
def load_model(model_name):
    try:
        if model_name == "GAT":
            weight_path = "weights/elliptic_gat.pt"
            ModelClass = GAT
            config_path = "configs/elliptic_gat.yaml"
        elif model_name == "GIN":
            weight_path = "weights/elliptic_gin.pt"
            ModelClass = GIN
            config_path = "configs/elliptic_gin.yaml"
        else: # GCN
            weight_path = "weights/elliptic_gcn.pt"
            ModelClass = GCN
            config_path = "configs/elliptic_gcn.yaml"

        conf = OmegaConf.load(config_path)
        
        # Paksa input dim ke 165 sesuai weights yang Anda miliki
        conf.model.input_dim = 165 
        
        model = ModelClass(conf.model).double()
        
        # Load Weights
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except RuntimeError as e:
        if "size mismatch" in str(e):
            st.error(f"⚠️ **Size Mismatch:** {e}")
        else:
            st.error(f"Runtime Error: {e}")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model(model_type)

# ==========================================
# 2. LOAD DATA
# ==========================================
@st.cache_resource
def load_lite_data():
    try:
        with open('data_lite.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Gagal load data: {e}")
        return None

data_lite = load_lite_data()

# ==========================================
# 3. INTERFACE PENGGUNA (MODIFIED)
# ==========================================
if data_lite and model:
    
    # --- MEMBUAT LIST URUTAN NODE ---
    # Mengambil semua key (TxID) dan mengubahnya menjadi list agar bisa diakses via index
    all_tx_ids = list(data_lite['features'].keys())
    total_nodes = len(all_tx_ids)
    
    st.divider()
    
    # Layout Kolom
    col_input, col_result = st.columns([1, 2])

    with col_input:
        st.subheader("🔢 Pilih Node")
        st.info(f"Total Transaksi tersedia: **{total_nodes}**")
        
        # INPUT BERDASARKAN NOMOR URUT (1 - Total)
        node_index = st.number_input(
            "Masukkan Nomor Urut Node:", 
            min_value=1, 
            max_value=total_nodes, 
            value=1,
            step=1
        )
        
        # Konversi Nomor Urut -> Transaction ID Asli
        # Dikurangi 1 karena list python mulai dari 0
        selected_tx_id = all_tx_ids[node_index - 1]
        
        st.caption(f"🆔 Original ID: `{selected_tx_id}`")
        
        analyze_btn = st.button("🔍 Analisa Node Ini", type="primary")
        
        # Optional: Tampilkan info status asli (jika ada di data untuk validasi)
        # Cek apakah ID ini termasuk illicit di data sample
        is_known_illicit = "Unknown"
        if 'illicit_examples' in data_lite:
            if selected_tx_id in data_lite['illicit_examples']:
                st.warning("Info Data Asli: Labelled Illicit")
            else:
                st.success("Info Data Asli: Labelled Licit/Unknown")

    with col_result:
        if analyze_btn:
            tx_id = selected_tx_id # Gunakan ID hasil mapping
            
            try:
                # --- PER
