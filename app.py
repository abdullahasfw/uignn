import streamlit as st
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.data import Data

# Import model definition dari project kamu
# Pastikan file models.py ada di folder fraud_detection
from fraud_detection.models import GAT, GCN, GIN

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Bitcoin Fraud Detection",
    page_icon="🕵️",
    layout="wide"
)

st.title("🕵️ Bitcoin Fraud Detection (Lite Version)")
st.markdown("Menggunakan **Graph Neural Networks (GNN)** untuk mendeteksi transaksi ilegal (Money Laundering) pada jaringan Bitcoin.")

# ==========================================
# 1. SETUP SIDEBAR & MODEL LOADING
# ==========================================
st.sidebar.header("⚙️ Konfigurasi Model")

model_type = st.sidebar.selectbox(
    "Pilih Arsitektur Model:",
    ("GAT", "GIN", "GCN")
)

@st.cache_resource
def load_model(model_name):
    """Load model weights tanpa perlu config file dataset"""
    try:
        # Tentukan path dan Class model
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

        # Load Config dummy untuk parameter model
        conf = OmegaConf.load(config_path)
        conf.model.input_dim = 166 # Hardcode input dim sesuai dataset Elliptic
        
        # Inisialisasi Model
        model = ModelClass(conf.model).double()
        
        # Load Weights
        # map_location='cpu' penting untuk deployment tanpa GPU
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Gagal memuat model {model_name}: {e}")
        return None

# Load model yang dipilih user
model = load_model(model_type)

# ==========================================
# 2. LOAD DATA LITE (PICKLE)
# ==========================================
@st.cache_resource
def load_lite_data():
    """Load sampel data yang sudah diringkas (features & adjacency)"""
    try:
        with open('data_lite.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except FileNotFoundError:
        st.error("❌ File 'data_lite.pkl' tidak ditemukan. Pastikan file ini ada di folder utama.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading pickle: {e}")
        return None

data_lite = load_lite_data()

# ==========================================
# 3. INTERFACE & PREDIKSI
# ==========================================

if data_lite is None:
    st.warning("⚠️ Data tidak tersedia. Mohon upload 'data_lite.pkl'.")
    st.stop()

if model is None:
    st.warning("⚠️ Model weights tidak ditemukan di folder 'weights/'.")
    st.stop()

# --- INPUT USER ---
st.divider()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔍 Cek Transaksi")
    
    # Ambil contoh ID Illicit dari pickle untuk memudahkan user
    example_id = data_lite['illicit_examples'][0] if 'illicit_examples' in data_lite else "Masukkan ID"
    st.info(f"Contoh ID Transaksi Ilegal: `{example_id}`")
    
    tx_id_input = st.text_input("Masukkan Transaction ID (Node ID):", value=str(example_id))
    analyze_btn = st.button("Analisa Transaksi", type="primary")

with col2:
    if analyze_btn:
        try:
            tx_id = int(tx_id_input)
            
            # 1. Cek apakah ID ada di data_lite
            if tx_id in data_lite['features']:
                
                # --- PERSIAPAN DATA GRAPH (MINI BATCH) ---
                # Ambil fitur target
                target_features = data_lite['features'][tx_id]
                
                # Ambil tetangga (neighbors) dari adjacency list
                neighbor_ids = data_lite['adj'].get(tx_id, [])
                
                # Kumpulkan node yang terlibat: [Target, Neighbor1, Neighbor2, ...]
                # Kita filter hanya neighbor yang punya fitur di data_lite
                valid_neighbors = [n for n in neighbor_ids if n in data_lite['features']]
                all_nodes = [tx_id] + valid_neighbors
                
                # Mapping ID Asli -> Index 0, 1, 2...
                node_map = {original_id: idx for idx, original_id in enumerate(all_nodes)}
                
                # Susun Matrix Features (X)
                x_list = [data_lite['features'][nid] for nid in all_nodes]
                x = torch.tensor(x_list, dtype=torch.double) # Shape: [Num_Nodes, 166]
                
                # Susun Edge Index
                edge_source = []
                edge_target = []
                
                target_idx = node_map[tx_id] # Pasti 0
                
                # Buat koneksi Target <-> Neighbors
                for neighbor in valid_neighbors:
                    n_idx = node_map[neighbor]
                    # Edge dua arah (Undirected)
                    edge_source.extend([target_idx, n_idx])
                    edge_target.extend([n_idx, target_idx])
                
                # Tambahkan Self-Loops (Penting untuk GNN)
                for i in range(len(all_nodes)):
                    edge_source.append(i)
                    edge_target.append(i)
                
                edge_index = torch.tensor([edge_source, edge_target], dtype=torch.long)
                
                # --- JALANKAN MODEL ---
                # Bungkus dalam Data object PyG
                mini_data = Data(x=x, edge_index=edge_index)
                
                with st.spinner("Sedang menganalisis pola transaksi..."):
                    with torch.no_grad():
                        out = model(mini_data)
                        
                        # Output model biasanya shape [N, 1] atau [N, 2]. 
                        # Kita ambil prediksi untuk index 0 (Target Node)
                        target_out = out[0]
                        
                        # Hitung probabilitas
                        if out.shape[-1] == 1:
                            prob = torch.sigmoid(target_out).item()
                        else:
                            # Jika output 2 class (logits)
                            prob = torch.softmax(target_out, dim=0)[1].item()
                
                # --- HASIL ---
                st.subheader("Hasil Analisis")
                m1, m2, m3 = st.columns(3)
                m1.metric("Transaction ID", tx_id)
                m2.metric("Risk Score", f"{prob:.2%}")
                
                prediction = 1 if prob > 0.5 else 0
                
                if prediction == 1:
                    m3.error("ILLICIT / FRAUD")
                    st.error(f"🚨 Transaksi ini memiliki probabilitas **{(prob*100):.2f}%** sebagai transaksi ilegal (Money Laundering).")
                else:
                    m3.success("LICIT / AMAN")
                    st.success(f"✅ Transaksi ini terdeteksi AMAN (Licit).")

                # --- VISUALISASI ---
                st.divider()
                st.subheader("🕸️ Visualisasi Tetangga")
                
                G_vis = nx.Graph()
                # Node utama
                color_map = []
                node_sizes = []
                
                # Tambah Target Node
                G_vis.add_node(tx_id)
                color_map.append('red' if prediction == 1 else 'green')
                node_sizes.append(600)
                
                # Tambah Neighbor Nodes
                for n_id in valid_neighbors:
                    G_vis.add_node(n_id)
                    G_vis.add_edge(tx_id, n_id)
                    color_map.append('lightgray')
                    node_sizes.append(200)
                
                fig, ax = plt.subplots(figsize=(8, 5))
                pos = nx.spring_layout(G_vis, seed=42)
                nx.draw(G_vis, pos, ax=ax, node_color=color_map, node_size=node_sizes, with_labels=False)
                # Label khusus untuk target
                nx.draw_networkx_labels(G_vis, pos, labels={tx_id: str(tx_id)}, font_color='white', font_size=8)
                
                st.pyplot(fig)
                
            else:
                st.warning(f"ID {tx_id} tidak ditemukan di dalam sampel data `data_lite.pkl`.")
                st.info("Cobalah gunakan ID contoh yang tersedia di atas.")
                
        except ValueError:
            st.error("Harap masukkan ID berupa angka.")
        except Exception as e:
            st.error(f"Terjadi error saat proses: {e}")
