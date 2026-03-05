import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.graph_objects as go
from datetime import datetime
import logging

# ==========================================
# KONFIGURASI HALAMAN PENGGUNA (UI)
# ==========================================
st.set_page_config(
    page_title="HydroData Processor Pro",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# KELAS PEMPROSESAN DATA (BACKEND ENGINE)
# ==========================================
class HydroDataProcessor:
    def __init__(self):
        # Gunakan in-memory string untuk menyimpan audit log supaya boleh dimuat turun
        self.log_stream = io.StringIO()
        self.logger = logging.getLogger("HydroLog")
        self.logger.setLevel(logging.INFO)
        
        # Elakkan duplicate handlers dalam Streamlit
        if not self.logger.handlers:
            handler = logging.StreamHandler(self.log_stream)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.logger.info("Sesi HydroData Processor dimulakan.")

    def parse_file(self, uploaded_file):
        """Modul A: Pengesan dan Pembaca Pelbagai Format Fail"""
        filename = uploaded_file.name.lower()
        self.logger.info(f"Membaca fail: {filename}")
        
        try:
            # Baca kandungan fail sebagai teks
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            
            # 1. Format XML (Data RLS)
            if "<VT t=" in content:
                self.logger.info("Format Dikesan: RLS XML")
                data = re.findall(r'<VT t="([^"]+)">([^<]+)</VT>', content)
                df = pd.DataFrame(data, columns=['timestamp', 'value'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df
                
            # 2. Format CSV/TXT RLS (Biasa) dengan sengkang '---'
            elif "---" in content[:500] or ".csv" in filename or ".txt" in filename:
                self.logger.info("Format Dikesan: RLS Text/CSV Standard")
                # Baca menggunakan pandas, kenal pasti '---' sebagai NaN
                uploaded_file.seek(0)
                # Menggunakan regex separator untuk pisahkan tarikh masa dan nilai
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, 
                                 names=['date', 'time', 'value'], na_values=['---'])
                # Gabungkan date dan time menjadi timestamp
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
                    df = df[['timestamp', 'value']]
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df
                
            # 3. Format DFT (.001)
            elif ".001" in filename:
                self.logger.info("Format Dikesan: DFT .001")
                lines = content.splitlines()
                # Anggap baris data bermula selepas baris yang mempunyai tanda -999 atau format tertentu
                raw_values = []
                for line in lines:
                    line = line.strip()
                    # Hanya ambil baris yang merupakan nombor murni atau nombor negatif
                    if re.match(r'^-?\d+$', line):
                        raw_values.append(float(line) / 100.0) # Contoh penukaran skala jika perlu
                
                # Cipta dummy timestamp untuk keperluan resampling (kerana .001 asal mungkin kompleks)
                if raw_values:
                    timestamps = pd.date_range(start=datetime.now().replace(hour=0, minute=0, second=0), 
                                               periods=len(raw_values), freq='10S')
                    df = pd.DataFrame({'timestamp': timestamps, 'value': raw_values})
                    return df
                else:
                    raise ValueError("Gagal mengekstrak data nilai dari fail .001")
                    
            else:
                st.error("Format fail tidak disokong atau tidak dikenali.")
                return None
                
        except Exception as e:
            self.logger.error(f"Ralat semasa membaca fail: {str(e)}")
            st.error(f"Ralat Pembacaan: {str(e)}")
            return None

    def process_data(self, df, threshold=9.0, method='linear', resample_rate='50S'):
        """Modul B: Pembersihan, Outlier Filtering & Resampling"""
        self.logger.info(f"Parameter Pemprosesan - Threshold: ±{threshold}, Interpolasi: {method}, Sela Masa: {resample_rate}")
        
        df_processed = df.copy()
        initial_nans = df_processed['value'].isna().sum()
        
        # 1. Kenal pasti dan buang outlier
        outliers_mask = df_processed['value'].abs() > threshold
        outliers_count = outliers_mask.sum()
        df_processed.loc[outliers_mask, 'value'] = np.nan
        self.logger.info(f"Dikesan {outliers_count} nilai outlier (melebihi julat ±{threshold}).")
        
        # 2. Interpolasi (Membaiki Missing Data & Outliers)
        total_missing = df_processed['value'].isna().sum()
        if total_missing > 0:
             # Elakkan ralat jika polynomial dipilih tetapi titik data tak cukup
            try:
                if method == 'polynomial':
                    df_processed['value'] = df_processed['value'].interpolate(method='polynomial', order=2)
                else:
                    df_processed['value'] = df_processed['value'].interpolate(method=method)
                
                # Fill baki NaN di hujung/awal graf menggunakan pad/bfill
                df_processed['value'] = df_processed['value'].ffill().bfill()
                self.logger.info(f"Berjaya membaiki {total_missing} data hilang/outlier dengan kaedah '{method}'.")
            except Exception as e:
                self.logger.warning(f"Ralat interpolasi {method}. Menukar ke 'linear'. Ralat: {str(e)}")
                df_processed['value'] = df_processed['value'].interpolate(method='linear').ffill().bfill()
        
        # 3. Resampling Data (Pengagregatan Masa)
        if resample_rate != 'Tiada':
            df_processed.set_index('timestamp', inplace=True)
            # Resample dan puratakan
            df_processed = df_processed.resample(resample_rate).mean()
            df_processed.reset_index(inplace=True)
            self.logger.info(f"Data berjaya diubah (resampled) ke sela masa {resample_rate}.")
            
        return df_processed, outliers_count, total_missing

    def get_audit_log(self):
        return self.log_stream.getvalue()


# ==========================================
# PEMBINAAN UI (STREAMLIT APP)
# ==========================================
def main():
    st.title("🌊 HydroData Processor Pro")
    st.markdown("Aplikasi Automasi Pintar untuk Pemprosesan Data Hidrologi (DFT & RLS). Menyokong format `.001`, `.txt` (XML), `.csv`.")

    # Inisialisasi Processor
    processor = HydroDataProcessor()

    # --- SIDEBAR (PANEL KAWALAN) ---
    with st.sidebar:
        st.header("⚙️ Panel Kawalan")
        uploaded_file = st.file_uploader("1. Muat Naik Fail Data", type=['001', 'txt', 'csv'])
        
        st.markdown("---")
        st.subheader("2. Parameter Penapisan (Filtering)")
        threshold = st.number_input("Threshold Nilai Oulier (±)", value=9.0, step=0.5, 
                                    help="Nilai melebihi tahap ini akan dibuang dan dikira sebagai outlier.")
        
        interpolation_method = st.selectbox("Kaedah Ganti Data (Interpolation)", 
                                            options=['linear', 'polynomial', 'pad', 'nearest'],
                                            help="Algoritma untuk mengisi kawasan data yang hilang atau telah dibuang.")
        
        st.markdown("---")
        st.subheader("3. Tetapan Sela Masa (Resampling)")
        resample_dict = {
            "Kekalkan Asal": "Tiada",
            "50 Saat (50S)": "50S",
            "1 Minit (1T)": "1min",
            "1 Jam (1H)": "1H"
        }
        resample_choice = st.selectbox("Format Masa Output", options=list(resample_dict.keys()))
        resample_val = resample_dict[resample_choice]

        st.markdown("---")
        process_btn = st.button("🚀 PROSES DATA SEKARANG", use_container_width=True, type="primary")

    # --- MAIN VIEW (RUANG PAPARAN) ---
    if uploaded_file is not None:
        # Pengekstrakan Data Mentah (Hanya berlaku sekali semasa muat naik)
        if 'raw_df' not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.raw_df = processor.parse_file(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
        
        raw_df = st.session_state.raw_df
        
        if raw_df is not None:
            # Gunakan Tab untuk kemasan antaramuka
            tab1, tab2, tab3 = st.tabs(["📊 Paparan Data", "📈 Visualisasi Interaktif", "📝 Audit Log & Muat Turun"])
            
            # --- TAB 1: DATA MENTAH ---
            with tab1:
                st.subheader(f"Informasi Fail: `{uploaded_file.name}`")
                col1, col2, col3 = st.columns(3)
                col1.metric("Jumlah Titik Data Asal", len(raw_df))
                col2.metric("Nilai Maksimum Asal", f"{raw_df['value'].max():.2f}" if not raw_df['value'].isna().all() else "N/A")
                col3.metric("Data Hilang / Kosong (NaN)", raw_df['value'].isna().sum())
                
                st.dataframe(raw_df.head(100), use_container_width=True)
                st.caption("Memaparkan 100 baris pertama data mentah.")

            # --- APABILA BUTANG PROSES DITEKAN ---
            if process_btn:
                with st.spinner('Memproses data, sedang menjalankan enjin penapisan...'):
                    # Proses Data
                    clean_df, out_count, missing_count = processor.process_data(
                        raw_df, 
                        threshold=threshold, 
                        method=interpolation_method, 
                        resample_rate=resample_val
                    )
                    
                    # Simpan hasil dalam session state
                    st.session_state.clean_df = clean_df
                    st.session_state.audit_log = processor.get_audit_log()
                    st.session_state.out_count = out_count
                    st.session_state.missing_count = missing_count
                    
            # --- TAB 2 & 3 (Hanya dipaparkan selepas data diproses) ---
            if 'clean_df' in st.session_state:
                clean_df = st.session_state.clean_df
                
                with tab2:
                    st.subheader("Perbandingan Sebelum & Selepas Pembersihan")
                    
                    # Metrik Ringkas
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Outlier Dibuang", st.session_state.out_count)
                    mc2.metric("Data Dibaiki (Interpolasi)", st.session_state.missing_count)
                    mc3.metric("Jumlah Baris Data Baru", len(clean_df))
                    
                    # Graf Interaktif menggunakan Plotly
                    fig = go.Figure()
                    # Graf Asal
                    fig.add_trace(go.Scatter(x=raw_df['timestamp'], y=raw_df['value'], 
                                             mode='lines', name='Data Asal (Raw)', 
                                             line=dict(color='red', width=1, dash='dot')))
                    # Graf Bersih
                    fig.add_trace(go.Scatter(x=clean_df['timestamp'], y=clean_df['value'], 
                                             mode='lines', name='Data Telah Diproses', 
                                             line=dict(color='blue', width=2)))
                    
                    fig.update_layout(title="Analisis Kualiti Data", 
                                      xaxis_title="Masa", yaxis_title="Nilai Bacaan (m)",
                                      hovermode="x unified", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("💡 Tip: Anda boleh zoom-in pada graf di atas untuk melihat bagaimana algoritma membetulkan titik data yang hilang/rosak.")

                with tab3:
                    st.subheader("Data Sedia Dieksport")
                    st.dataframe(clean_df.head(100), use_container_width=True)
                    
                    st.subheader("Audit Trail System")
                    st.text_area("Log Pindaan & Proses", value=st.session_state.audit_log, height=200)
                    
                    st.markdown("---")
                    st.subheader("📥 Muat Turun Hasil")
                    col_dl1, col_dl2 = st.columns(2)
                    
                    # Convert DF to CSV untuk muat turun
                    csv = clean_df.to_csv(index=False).encode('utf-8')
                    col_dl1.download_button(
                        label="⬇️ Muat Turun Data Bersih (.CSV)",
                        data=csv,
                        file_name=f"Processed_{uploaded_file.name}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                    # Muat turun Log
                    log_data = st.session_state.audit_log.encode('utf-8')
                    col_dl2.download_button(
                        label="⬇️ Muat Turun Audit Log (.TXT)",
                        data=log_data,
                        file_name=f"AuditLog_{uploaded_file.name}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
    else:
        # Jika belum muat naik
        st.info("Sila muat naik fail data anda dari panel di sebelah kiri untuk bermula.")
        
if __name__ == "__main__":
    main()
