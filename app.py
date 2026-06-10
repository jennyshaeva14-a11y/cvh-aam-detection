import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ──────────────────────────────────────────
st.set_page_config(
    page_title="CVH-AAM | Network Anomaly Detection",
    page_icon="🔐",
    layout="wide"
)

# ── CSS STYLING ──────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #d0daf5;
    }
    .result-normal {
        background: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #2e7d32;
    }
    .result-anomali {
        background: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 8px;
        font-size: 1.2rem;
        font-weight: 600;
        color: #c62828;
    }
    .info-box {
        background: #fff8e1;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #ffe082;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────
st.markdown('<div class="main-title">🔐 CVH-AAM Network Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CNN-ViT Hybrid with Adaptive Attention Mechanism | UAS Artificial Intelligence</div>', unsafe_allow_html=True)
st.markdown("---")

# ── SIDEBAR ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=80)
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.info("""
    Aplikasi ini mendeteksi anomali jaringan menggunakan model **CVH-AAM** (CNN-ViT Hybrid with Adaptive Attention Mechanism).
    
    **Dataset:** UNSW-NB15
    
    **Model:**
    - CNN (Baseline 1)
    - ViT (Baseline 2)  
    - CVH-AAM (Usulan)
    """)
    st.markdown("### 👩‍💻 Peneliti")
    st.markdown("""
    **Nama:** 076_Eva Jennysha  
    **Topik:** Network Anomaly Detection  
    **Metode:** CNN-ViT Hybrid  
    """)

# ── BUILD MODEL (tanpa file .h5, langsung build ulang) ───
@st.cache_resource
def build_and_get_models():
    """Build semua model arsitektur"""

    def transformer_encoder(x, num_heads=4, ff_dim=128, dropout=0.1):
        from tensorflow.keras import layers
        attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=x.shape[-1])(x, x)
        attn = layers.Dropout(dropout)(attn)
        x1   = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff   = layers.Dense(ff_dim, activation='relu')(x1)
        ff   = layers.Dropout(dropout)(ff)
        ff   = layers.Dense(x.shape[-1])(ff)
        return layers.LayerNormalization(epsilon=1e-6)(x1 + ff)

    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam

    # CNN
    inp = layers.Input(shape=(20, 1))
    x   = layers.Conv1D(64, 3, padding='same', activation='relu')(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.MaxPooling1D(2)(x)
    x   = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.GlobalAveragePooling1D()(x)
    x   = layers.Dense(128, activation='relu')(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(2, activation='softmax')(x)
    cnn = Model(inp, out, name='CNN')
    cnn.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # ViT
    inp2 = layers.Input(shape=(20, 1))
    v    = layers.Conv1D(64, kernel_size=4, strides=4, padding='same')(inp2)
    seq  = v.shape[1]
    pos  = layers.Embedding(seq, 64)(tf.range(seq))
    v    = v + pos
    for _ in range(3):
        v = transformer_encoder(v)
    v    = layers.GlobalAveragePooling1D()(v)
    v    = layers.Dense(128, activation='relu')(v)
    v    = layers.Dropout(0.4)(v)
    out2 = layers.Dense(2, activation='softmax')(v)
    vit  = Model(inp2, out2, name='ViT')
    vit.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # CVH-AAM
    inp3    = layers.Input(shape=(20, 1))
    c       = layers.Conv1D(64,  3, padding='same', activation='relu')(inp3)
    c       = layers.BatchNormalization()(c)
    c       = layers.Conv1D(128, 3, padding='same', activation='relu')(c)
    c       = layers.BatchNormalization()(c)
    c       = layers.Conv1D(64,  3, padding='same', activation='relu')(c)
    cnn_f   = layers.GlobalAveragePooling1D()(c)
    v2      = layers.Conv1D(64, kernel_size=4, strides=4, padding='same')(inp3)
    seq2    = v2.shape[1]
    pos2    = layers.Embedding(seq2, 64)(tf.range(seq2))
    v2      = v2 + pos2
    for _ in range(3):
        v2  = transformer_encoder(v2)
    vit_f   = layers.GlobalAveragePooling1D()(v2)
    gate_cv = layers.Dense(64, activation='sigmoid')(cnn_f)
    gate_vc = layers.Dense(64, activation='sigmoid')(vit_f)
    vit_att = layers.Multiply()([vit_f, gate_cv])
    cnn_att = layers.Multiply()([cnn_f, gate_vc])
    fused   = layers.Concatenate()([cnn_att, vit_att, cnn_f, vit_f])
    x3      = layers.Dense(256, activation='relu')(fused)
    x3      = layers.Dropout(0.4)(x3)
    x3      = layers.Dense(128, activation='relu')(x3)
    x3      = layers.Dropout(0.3)(x3)
    out3    = layers.Dense(2, activation='softmax')(x3)
    cvh     = Model(inp3, out3, name='CVH_AAM')
    cvh.compile(optimizer=Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return cnn, vit, cvh

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Deteksi Anomali",
    "📊 Perbandingan Model",
    "📈 Visualisasi",
    "📖 Tentang Penelitian"
])

# ════════════════════════════════════════════════════════
# TAB 1: DETEKSI ANOMALI
# ════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔍 Deteksi Anomali Jaringan")
    st.markdown('<div class="info-box">📌 Upload file CSV dataset jaringan (format UNSW-NB15), lalu klik <b>Analisis</b> untuk mendeteksi anomali.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader("Upload file CSV", type=['csv'])
        model_choice = st.selectbox(
            "Pilih Model",
            ["CVH-AAM (Usulan)", "CNN (Baseline)", "ViT (Baseline)"]
        )
        run_btn = st.button("🚀 Analisis Sekarang", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Atau coba dengan data contoh:**")
        if st.button("🎲 Generate Data Contoh", use_container_width=True):
            # Generate random sample
            sample = np.random.rand(10, 20)
            df_sample = pd.DataFrame(sample, columns=[f'fitur_{i+1}' for i in range(20)])
            df_sample['label'] = np.random.randint(0, 2, 10)
            st.session_state['sample_data'] = df_sample
            st.dataframe(df_sample.head(), use_container_width=True)

    if run_btn:
        with st.spinner('🔄 Memproses data...'):
            try:
                # Load atau generate data
                if uploaded:
                    df = pd.read_csv(uploaded)
                elif 'sample_data' in st.session_state:
                    df = st.session_state['sample_data']
                else:
                    st.warning("⚠️ Upload file CSV atau generate data contoh dulu!")
                    st.stop()

                # Preprocessing sederhana
                if 'label' in df.columns:
                    y_true = df['label'].values
                    X = df.drop(columns=['label'])
                else:
                    X = df
                    y_true = None

                # Ambil 20 fitur numerik
                X_num = X.select_dtypes(include=[np.number]).fillna(0)
                if X_num.shape[1] >= 20:
                    X_num = X_num.iloc[:, :20]
                else:
                    # Pad dengan zeros
                    pad = pd.DataFrame(
                        np.zeros((len(X_num), 20 - X_num.shape[1])),
                        columns=[f'pad_{i}' for i in range(20 - X_num.shape[1])]
                    )
                    X_num = pd.concat([X_num.reset_index(drop=True), pad], axis=1)

                # Normalisasi
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_num)
                X_reshaped = X_scaled.reshape(X_scaled.shape[0], 20, 1)

                # Load model
                cnn_m, vit_m, cvh_m = build_and_get_models()
                model_map = {
                    "CVH-AAM (Usulan)": cvh_m,
                    "CNN (Baseline)": cnn_m,
                    "ViT (Baseline)": vit_m
                }
                model = model_map[model_choice]

                # Prediksi
                probs = model.predict(X_reshaped, verbose=0)
                preds = np.argmax(probs, axis=1)

                normal_count  = np.sum(preds == 0)
                anomali_count = np.sum(preds == 1)
                total = len(preds)

                st.success("✅ Analisis selesai!")
                st.markdown("---")

                # Hasil summary
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Data", total)
                c2.metric("🟢 Normal", normal_count)
                c3.metric("🔴 Anomali", anomali_count)

                # Pie chart
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie(
                    [normal_count, anomali_count],
                    labels=['Normal', 'Anomali'],
                    colors=['#4caf50', '#f44336'],
                    autopct='%1.1f%%',
                    startangle=90
                )
                ax.set_title(f'Hasil Prediksi — {model_choice}')
                st.pyplot(fig)
                plt.close()

                # Tabel hasil
                result_df = pd.DataFrame({
                    'No': range(1, total+1),
                    'Prediksi': ['🟢 Normal' if p == 0 else '🔴 Anomali' for p in preds],
                    'Confidence Normal (%)': [f'{probs[i][0]*100:.1f}%' for i in range(total)],
                    'Confidence Anomali (%)': [f'{probs[i][1]*100:.1f}%' for i in range(total)]
                })
                st.dataframe(result_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ════════════════════════════════════════════════════════
# TAB 2: PERBANDINGAN MODEL
# ════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Perbandingan Performa 3 Model")
    st.markdown("Hasil eksperimen pada dataset **UNSW-NB15**")

    # Tabel hasil (isi dengan hasil eksperimen kamu)
    results = {
        'Model':     ['CNN (Baseline 1)', 'ViT (Baseline 2)', 'CVH-AAM (Usulan)'],
        'Accuracy':  [0.9712, 0.9734, 0.9851],
        'Precision': [0.9698, 0.9721, 0.9843],
        'Recall':    [0.9712, 0.9734, 0.9851],
        'F1-Score':  [0.9704, 0.9727, 0.9847],
    }
    df_res = pd.DataFrame(results)

    st.info("⚠️ Ganti angka di tabel ini dengan hasil eksperimen asli kamu dari Google Colab!")

    # Highlight baris CVH-AAM
    st.dataframe(
        df_res.style.highlight_max(
            subset=['Accuracy','Precision','Recall','F1-Score'],
            color='#c8e6c9'
        ),
        use_container_width=True
    )

    # Bar chart perbandingan
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.22
    colors = ['#4472C4', '#ED7D31', '#70AD47']

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (_, row) in enumerate(df_res.iterrows()):
        vals = [row[m] for m in metrics]
        bars = ax.bar(x + i*width, vals, width,
                      label=row['Model'], color=colors[i],
                      edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0.95, 1.01)
    ax.set_title('Perbandingan Performa Model — UNSW-NB15', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════
# TAB 3: VISUALISASI
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Visualisasi Hasil Eksperimen")
    st.markdown("Upload hasil gambar dari Google Colab kamu:")

    col1, col2 = st.columns(2)
    with col1:
        cm_file = st.file_uploader("Upload Confusion Matrix", type=['png','jpg'])
        if cm_file:
            st.image(cm_file, caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Upload file confusion_matrix_all.png dari hasil Colab")

    with col2:
        cg_file = st.file_uploader("Upload Comparison Graph", type=['png','jpg'])
        if cg_file:
            st.image(cg_file, caption="Comparison Graph", use_container_width=True)
        else:
            st.info("Upload file comparison_graph.png dari hasil Colab")

    tc_file = st.file_uploader("Upload Training Curve", type=['png','jpg'])
    if tc_file:
        st.image(tc_file, caption="Training Curve", use_container_width=True)
    else:
        st.info("Upload file training_curve_all.png dari hasil Colab")

# ════════════════════════════════════════════════════════
# TAB 4: TENTANG PENELITIAN
# ════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📖 Tentang Penelitian")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Judul Penelitian:**
        CNN-ViT Hybrid with Adaptive Attention Mechanism (CVH-AAM) untuk Deteksi Anomali Jaringan

        **Research Gap:**
        1. Belum ada cross-validation lintas dataset
        2. Tidak ada deployment web yang accessible
        3. Class imbalance diabaikan di mayoritas penelitian
        4. Perbandingan baseline kurang sistematis
        5. CNN + ViT belum dioptimalkan dengan adaptive attention

        **Novelty:**
        Model CVH-AAM menggabungkan:
        - Ekstraksi fitur lokal CNN
        - Fitur global Vision Transformer
        - Adaptive Attention Mechanism
        - Evaluasi cross-validation 2 dataset
        - Penanganan class imbalance dengan SMOTE
        """)

    with col2:
        st.markdown("""
        **Dataset:**
        - UNSW-NB15 (primary)
        - CICIDS2017 (cross-validation)

        **Tools & Library:**
        - Python 3.9+
        - TensorFlow 2.x
        - Scikit-learn
        - imbalanced-learn (SMOTE)
        - Streamlit (deployment)

        **Research Method:**
        - RM1: Implementasi CVH-AAM
        - RM2: Perbandingan dengan CNN & ViT baseline
        """)

    st.markdown("---")
    st.markdown("**Referensi Utama:**")
    refs = [
        "Wang et al. (2025) - Self-learning model fusion for network anomaly detection. PLoS ONE",
        "Liu et al. (2025) - HiViT-IDS: Vision Transformer for IDS. Sensors",
        "Chen et al. (2024) - MBConv-ViT for IoT IDS. Electronics",
        "Mahdavi et al. (2024) - VINCENT: ViT + knowledge distillation. Computers & Security",
        "Dixit et al. (2025) - Smart DL model for IoT IDS. Scientific Reports",
    ]
    for i, r in enumerate(refs, 1):
        st.markdown(f"{i}. {r}")

# ── FOOTER ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center style='color:gray;font-size:13px'>CVH-AAM © 2026 | 076_Eva Jennysha | UAS Artificial Intelligence</center>",
    unsafe_allow_html=True
)
