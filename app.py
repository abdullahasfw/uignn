import streamlit as st
import torch
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch_geometric
from torch_geometric.data import Data

# Import model definition
from fraud_detection.models import GAT, GCN, GIN

st.set_page_config(page_title="Bitcoin Fraud Detection", page_icon="🕵️", layout="wide")

st.title("🕵️ Bitcoin Fraud Detection (Lite Version)")
st.markdown("Menggunakan **Graph Neural Networks (GNN)** untuk mendeteksi transaksi ilegal.")

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
        
        # --- PERBAIKAN DI SINI ---
        # Ubah input_dim jadi 165 agar sesuai dengan weights (checkpoint)
        conf.model.input_dim = 165 
        
        model = ModelClass(conf.model).double()
        
        # Load Weights
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except RuntimeError as e:
        if "size mismatch" in str(e):
            st.error(f"⚠️ **Size Mismatch Error:** {e}")
            st.info("Sistem mencoba menyesuaikan dimensi input, tapi struktur weights berbeda.")
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
# 3. INTERFACE
# ==========================================
if data_lite and model:
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🔍 Cek Transaksi")
        # Ambil contoh ID yang valid
        if 'illicit_examples' in data_lite and len(data_lite['illicit_examples']) > 0:
            example_id = data_lite['illicit_examples'][0]
        else:
            # Fallback jika list kosong, ambil key pertama dari features
            example_id = list(data_lite['features'].keys())[0]

        st.info(f"Contoh ID Transaksi: `{example_id}`")
        tx_id_input = st.text_input("Transaction ID:", value=str(example_id))
        analyze_btn = st.button("Analisa", type="primary")

    with col2:
        if analyze_btn:
            try:
                tx_id = int(tx_id_input)
                if tx_id in data_lite['features']:
                    
                    # --- PERSIAPAN DATA ---
                    neighbor_ids = data_lite['adj'].get(tx_id, [])
                    valid_neighbors = [n for n in neighbor_ids if n in data_lite['features']]
                    all_nodes = [tx_id] + valid_neighbors
                    
                    node_map = {orig: i for i, orig in enumerate(all_nodes)}
                    
                    # Ambil features raw
                    x_list = [data_lite['features'][nid] for nid in all_nodes]
                    x = torch.tensor(x_list, dtype=torch.double)
                    
                    # --- PERBAIKAN DIMENSI FITUR ---
                    # Jika data punya 166 fitur tapi model minta 165, kita potong 1 terakhir (atau pertama)
                    # Biasanya time-step dibuang. Kita coba ambil 165 fitur pertama.
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

                    # --- PREDIKSI ---
                    with torch.no_grad():
                        out = model(mini_data)
                        target_out = out[0]
                        
                        if out.shape[-1] == 1:
                            prob = torch.sigmoid(target_out).item()
                        else:
                            prob = torch.softmax(target_out, dim=0)[1].item()
                    
                    # Tampilkan Hasil
                    st.subheader("Hasil Analisis")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Risk Score", f"{prob:.2%}")
                    
                    if prob > 0.5:
                        res_col2.error("🚨 ILLICIT (Fraud)")
                    else:
                        res_col2.success("✅ LICIT (Aman)")
                        
                    # Visualisasi Simple
                    st.caption(f"Visualisasi Graph (Target + {len(valid_neighbors)} Neighbors)")
                    G = nx.Graph()
                    G.add_node(tx_id, color='red' if prob > 0.5 else 'green')
                    for n in valid_neighbors:
                        G.add_node(n, color='gray')
                        G.add_edge(tx_id, n)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    colors = [nx.get_node_attributes(G, 'color')[n] for n in G.nodes()]
                    nx.draw(G, ax=ax, node_color=colors, with_labels=False, node_size=100)
                    st.pyplot(fig)

                else:
                    st.error("ID tidak ditemukan di database sampel.")
            except Exception as e:
                st.error(f"Error: {e}")
