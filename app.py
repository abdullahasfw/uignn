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
                # --- PERSIAPAN DATA GRAPH ---
                neighbor_ids = data_lite['adj'].get(tx_id, [])
                valid_neighbors = [n for n in neighbor_ids if n in data_lite['features']]
                all_nodes = [tx_id] + valid_neighbors
                
                node_map = {orig: i for i, orig in enumerate(all_nodes)}
                
                # Ambil features
                x_list = [data_lite['features'][nid] for nid in all_nodes]
                x = torch.tensor(x_list, dtype=torch.double)
                
                # Potong fitur jadi 165 (fix size mismatch)
                if x.shape[1] == 166:
                    x = x[:, :165] 
                
                # Buat Edge Index
                edge_src, edge_dst = [], []
                target_idx = 0 
                
                for neighbor in valid_neighbors:
                    n_idx = node_map[neighbor]
                    edge_src.extend([target_idx, n_idx])
                    edge_dst.extend([n_idx, target_idx])
                
                # Self loops
                for i in range(len(all_nodes)):
                    edge_src.append(i)
                    edge_dst.append(i)
                    
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
                mini_data = Data(x=x, edge_index=edge_index)

                # --- PREDIKSI MODEL ---
                with torch.no_grad():
                    out = model(mini_data)
                    target_out = out[0]
                    
                    if out.shape[-1] == 1:
                        prob = torch.sigmoid(target_out).item()
                    else:
                        prob = torch.softmax(target_out, dim=0)[1].item()
                
                # --- HASIL ---
                st.subheader(f"Hasil Analisis Node #{node_index}")
                
                res1, res2, res3 = st.columns(3)
                res1.metric("Node Index", f"#{node_index}")
                res2.metric("Original TxID", str(tx_id)[:8]+"...") # Singkat ID biar rapi
                
                prediction_label = "ILLICIT (FRAUD)" if prob > 0.5 else "LICIT (SAFE)"
                
                if prob > 0.5:
                    res3.metric("Prediction", prediction_label, delta=f"{prob:.1%}")
                    st.error(f"🚨 **PERINGATAN:** Model mendeteksi transaksi ini mencurigakan dengan probabilitas {prob:.4f}.")
                else:
                    res3.metric("Prediction", prediction_label, delta_color="normal")
                    st.success(f"✅ **AMAN:** Transaksi ini terlihat normal (Probabilitas Fraud: {prob:.4f}).")
                    
                # --- VISUALISASI ---
                st.divider()
                st.write("**Visualisasi Koneksi (Ego Graph)**")
                
                G = nx.Graph()
                # Node Pusat
                G.add_node(tx_id, color='red' if prob > 0.5 else 'green', size=800)
                
                # Node Tetangga
                for n in valid_neighbors:
                    G.add_node(n, color='lightgray', size=300)
                    G.add_edge(tx_id, n)
                
                fig, ax = plt.subplots(figsize=(7, 4))
                
                colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
                sizes = [nx.get_node_attributes(G, 'size')[n] for n in G.nodes()]
                
                # Layout spring agar menyebar
                pos = nx.spring_layout(G, seed=42)
                
                nx.draw(G, pos, ax=ax, node_color=colors, node_size=sizes, with_labels=False, edge_color='gray', alpha=0.7)
                
                # Label khusus node pusat (menggunakan No Urut agar mudah dibaca)
                nx.draw_networkx_labels(G, pos, labels={tx_id: f"#{node_index}"}, font_color='white', font_weight='bold')
                
                st.pyplot(fig)

            except Exception as e:
                st.error(f"Terjadi error: {e}")
