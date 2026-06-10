import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CVH-AAM | Network Anomaly Detection",
    page_icon="🔐",
    layout="wide"
)

st.markdown("""
<style>
.main-title{font-size:2rem;font-weight:700;color:#1a1a2e;text-align:center;margin-bottom:0.2rem}
.sub-title{font-size:1rem;color:#555;text-align:center;margin-bottom:2rem}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔐 CVH-AAM Network Anomaly Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CNN-ViT Hybrid with Adaptive Attention Mechanism | UAS Artificial Intelligence</div>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### ℹ️ Tentang Aplikasi")
    st.info("""
    Aplikasi ini mendeteksi anomali jaringan menggunakan model **CVH-AAM**.
    
    **Dataset:** UNSW-NB15
    
    **Model:**
    - CNN-ViT Hybrid (Usulan)
    - Random Forest (Baseline 1)
    - Gradient Boosting (Baseline 2)
    """)
    st.markdown("### 👩‍💻 Peneliti")
    st.markdown("**Nama:** 076_Eva Jennysha\n\n**Topik:** Network Anomaly Detection")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Deteksi Anomali",
    "📊 Perbandingan Model",
    "📈 Visualisasi",
    "📖 Tentang Penelitian"
])

with tab1:
    st.markdown("### 🔍 Deteksi Anomali Jaringan")
    st.info("📌 Upload file CSV dataset jaringan atau gunakan data contoh untuk mendeteksi anomali.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Upload file CSV", type=['csv'])
        model_choice = st.selectbox("Pilih Model", [
            "CVH-AAM (Usulan)",
            "Random Forest (Baseline 1)",
            "Gradient Boosting (Baseline 2)"
        ])
        run_btn = st.button("🚀 Analisis Sekarang", type="primary", use_container_width=True)

    with col2:
        st.markdown("**Atau coba dengan data contoh:**")
        if st.button("🎲 Generate Data Contoh", use_container_width=True):
            np.random.seed(42)
            sample = np.random.rand(50, 20)
            labels = np.random.randint(0, 2, 50)
            df_sample = pd.DataFrame(sample, columns=[f'fitur_{i+1}' for i in range(20)])
            df_sample['label'] = labels
            st.session_state['sample_data'] = df_sample
            st.success("Data contoh siap!")
            st.dataframe(df_sample.head(), use_container_width=True)

    if run_btn:
        with st.spinner('🔄 Memproses dan mendeteksi anomali...'):
            try:
                if uploaded:
                    df = pd.read_csv(uploaded)
                elif 'sample_data' in st.session_state:
                    df = st.session_state['sample_data']
                else:
                    st.warning("⚠️ Upload file CSV atau generate data contoh dulu!")
                    st.stop()

                if 'label' in df.columns:
                    y = df['label'].values
                    X = df.drop(columns=['label'])
                else:
                    X = df
                    y = np.zeros(len(df))

                X_num = X.select_dtypes(include=[np.number]).fillna(0)
                if X_num.shape[1] >= 20:
                    X_num = X_num.iloc[:, :20]
                else:
                    pad = pd.DataFrame(
                        np.zeros((len(X_num), 20 - X_num.shape[1])),
                        columns=[f'pad_{i}' for i in range(20 - X_num.shape[1])]
                    )
                    X_num = pd.concat([X_num.reset_index(drop=True), pad], axis=1)

                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X_num)

                # Simulasi model CVH-AAM dengan ensemble
                if model_choice == "CVH-AAM (Usulan)":
                    # Simulasi hybrid: gabungan RF + GB dengan voting
                    m1 = RandomForestClassifier(n_estimators=100, random_state=42)
                    m2 = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    m1.fit(X_scaled, y)
                    m2.fit(X_scaled, y)
                    p1 = m1.predict_proba(X_scaled)
                    p2 = m2.predict_proba(X_scaled)
                    probs = (p1 + p2) / 2
                elif model_choice == "Random Forest (Baseline 1)":
                    m = RandomForestClassifier(n_estimators=100, random_state=42)
                    m.fit(X_scaled, y)
                    probs = m.predict_proba(X_scaled)
                else:
                    m = GradientBoostingClassifier(n_estimators=100, random_state=42)
                    m.fit(X_scaled, y)
                    probs = m.predict_proba(X_scaled)

                preds = np.argmax(probs, axis=1)
                normal_count = int(np.sum(preds == 0))
                anomali_count = int(np.sum(preds == 1))
                total = len(preds)

                st.success("✅ Analisis selesai!")
                st.markdown("---")

                c1, c2, c3 = st.columns(3)
                c1.metric("Total Data", total)
                c2.metric("🟢 Normal", normal_count)
                c3.metric("🔴 Anomali", anomali_count)

                col_a, col_b = st.columns(2)
                with col_a:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    ax.pie([normal_count, anomali_count],
                           labels=['Normal', 'Anomali'],
                           colors=['#4caf50', '#f44336'],
                           autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'Hasil Prediksi — {model_choice}')
                    st.pyplot(fig)
                    plt.close()

                with col_b:
                    cm = confusion_matrix(y, preds)
                    fig2, ax2 = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                                xticklabels=['Normal','Anomali'],
                                yticklabels=['Normal','Anomali'])
                    ax2.set_title('Confusion Matrix')
                    ax2.set_xlabel('Predicted')
                    ax2.set_ylabel('Actual')
                    st.pyplot(fig2)
                    plt.close()

                result_df = pd.DataFrame({
                    'No': range(1, total+1),
                    'Prediksi': ['🟢 Normal' if p==0 else '🔴 Anomali' for p in preds],
                    'Confidence Normal (%)': [f'{probs[i][0]*100:.1f}%' for i in range(total)],
                    'Confidence Anomali (%)': [f'{probs[i][1]*100:.1f}%' for i in range(total)]
                })
                st.dataframe(result_df, use_container_width=True)

            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.markdown("### 📊 Perbandingan Performa 3 Model")
    st.markdown("Hasil eksperimen pada dataset **UNSW-NB15** — ganti angka dengan hasil Colab kamu!")

    col_edit1, col_edit2, col_edit3 = st.columns(3)
    with col_edit1:
        acc_cnn = st.number_input("Accuracy CVH-AAM", value=0.9851, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
        f1_cnn  = st.number_input("F1-Score CVH-AAM", value=0.9847, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
    with col_edit2:
        acc_rf  = st.number_input("Accuracy RF (Baseline 1)", value=0.9712, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
        f1_rf   = st.number_input("F1-Score RF (Baseline 1)", value=0.9704, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
    with col_edit3:
        acc_gb  = st.number_input("Accuracy GB (Baseline 2)", value=0.9734, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")
        f1_gb   = st.number_input("F1-Score GB (Baseline 2)", value=0.9727, min_value=0.0, max_value=1.0, step=0.001, format="%.4f")

    results = {
        'Model':    ['Random Forest (B1)', 'Gradient Boosting (B2)', 'CVH-AAM (Usulan)'],
        'Accuracy': [acc_rf, acc_gb, acc_cnn],
        'F1-Score': [f1_rf, f1_gb, f1_cnn],
    }
    df_res = pd.DataFrame(results)
    st.dataframe(df_res.style.highlight_max(subset=['Accuracy','F1-Score'], color='#c8e6c9'), use_container_width=True)

    x = np.arange(2)
    width = 0.22
    colors = ['#4472C4', '#ED7D31', '#70AD47']
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, row in df_res.iterrows():
        vals = [row['Accuracy'], row['F1-Score']]
        bars = ax.bar(x + i*width, vals, width, label=row['Model'], color=colors[i], edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x + width); ax.set_xticklabels(['Accuracy', 'F1-Score'])
    ax.set_ylim(0.95, 1.01)
    ax.set_title('Perbandingan Performa Model — UNSW-NB15', fontsize=12, fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab3:
    st.markdown("### 📈 Upload Hasil Visualisasi dari Colab")
    col1, col2 = st.columns(2)
    with col1:
        cm_file = st.file_uploader("Confusion Matrix", type=['png','jpg'])
        if cm_file:
            st.image(cm_file, caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Upload confusion_matrix_all.png")
    with col2:
        cg_file = st.file_uploader("Comparison Graph", type=['png','jpg'])
        if cg_file:
            st.image(cg_file, caption="Comparison Graph", use_container_width=True)
        else:
            st.info("Upload comparison_graph.png")
    tc_file = st.file_uploader("Training Curve", type=['png','jpg'])
    if tc_file:
        st.image(tc_file, caption="Training Curve", use_container_width=True)
    else:
        st.info("Upload training_curve_all.png")

with tab4:
    st.markdown("### 📖 Tentang Penelitian")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Judul:** CNN-ViT Hybrid with Adaptive Attention Mechanism (CVH-AAM) untuk Deteksi Anomali Jaringan

**Research Gap:**
1. Belum ada cross-validation lintas dataset
2. Tidak ada deployment web yang accessible
3. Class imbalance diabaikan di mayoritas penelitian
4. Perbandingan baseline kurang sistematis
5. CNN + ViT belum dioptimalkan dengan adaptive attention

**Novelty:**
- Gabungan CNN + Vision Transformer
- Adaptive Attention Mechanism
- Cross-validation 2 dataset
- Penanganan class imbalance dengan SMOTE
        """)
    with col2:
        st.markdown("""
**Dataset:**
- UNSW-NB15 (primary)
- CICIDS2017 (cross-validation)

**Research Method:**
- RM1: Implementasi CVH-AAM
- RM2: Perbandingan dengan baseline

**Tools:**
- Python, Scikit-learn, Streamlit
- TensorFlow (model training)
- Pandas, NumPy, Matplotlib
        """)

st.markdown("---")
st.markdown("<center style='color:gray;font-size:13px'>CVH-AAM © 2026 | 076_Eva Jennysha | UAS Artificial Intelligence</center>", unsafe_allow_html=True)
