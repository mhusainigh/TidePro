import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.graph_objects as go
from datetime import datetime
import logging
import time

st.set_page_config(page_title="HydroData Processor Pro", page_icon="🌊", layout="wide")

class HydroDataProcessor:
    def __init__(self):
        self.log_stream = io.StringIO()
        self.logger = logging.getLogger("HydroLog")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler(self.log_stream)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
            
    def parse_file(self, uploaded_file):
        """Modul A (V2): Parser Lebih Pintar untuk DFT, RLS, dan KUD"""
        filename = uploaded_file.name.lower()
        self.logger.info(f"Membaca fail: {filename}")
        
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            
            # 1. Format XML (Data RLS .txt)
            if "<VT t=" in content:
                content = content.replace('""', '"')
                data = re.findall(r'<VT t="([^"]+)">\s*([0-9.-]+)\s*</VT>', content)
                df = pd.DataFrame(data, columns=['timestamp', 'value'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df
                
            # 2. Format CSV/TXT RLS (Standard format dengan '---')
            elif "---" in content[:1000] or ".csv" in filename:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, 
                                 names=['date', 'time', 'value'], na_values=['---'])
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
                    df = df[['timestamp', 'value']]
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.dropna(subset=['timestamp'])
                
            # 3. Format DFT (.001)
            elif ".001" in filename:
                lines = content.splitlines()
                start_time = None
                raw_values = []
                start_reading = False
                
                for line in lines:
                    line = line.strip()
                    if not start_reading:
                        match = re.search(r'(\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                        if match:
                            start_time = pd.to_datetime(match.group(1))
                            start_reading = True
                    else:
                        if re.match(r'^-?\d+$', line):
                            val = float(line)
                            if val == -999:
                                raw_values.append(np.nan)
                            else:
                                raw_values.append(val / 100.0)
                
                if start_time is not None and len(raw_values) > 0:
                    timestamps = pd.date_range(start=start_time, periods=len(raw_values), freq='10S')
                    return pd.DataFrame({'timestamp': timestamps, 'value': raw_values})
                else:
                    raise ValueError("Gagal mengesan masa bermula (timestamp) dalam fail .001")

            # 4. Format .kud
            elif ".kud" in filename:
                lines = content.splitlines()
                raw_values = []
                for line in lines[50:]:
                    line = line.strip()
                    if re.match(r'^-?\d+$', line):
                        raw_values.append(float(line) / 100.0)
                
                timestamps = pd.date_range(start=datetime.now().replace(hour=0, minute=0, second=0), periods=len(raw_values), freq='50S')
                return pd.DataFrame({'timestamp': timestamps, 'value': raw_values})
                    
            else:
                st.error("Format fail tidak disokong.")
                return None
                
        except Exception as e:
            st.error(f"Ralat Pembacaan Modul: {str(e)}")
            return None

    def process_data(self, df, threshold=9.0, method='linear', resample_rate='50S'):
        df_processed = df.copy()
        outliers_mask = df_processed['value'].abs() > threshold
        outliers_count = outliers_mask.sum()
        df_processed.loc[outliers_mask, 'value'] = np.nan
        
        total_missing = df_processed['value'].isna().sum()
        if total_missing > 0:
            try:
                if method == 'polynomial':
                    df_processed['value'] = df_processed['value'].interpolate(method='polynomial', order=2)
                else:
                    df_processed['value'] = df_processed['value'].interpolate(method=method)
                df_processed['value'] = df_processed['value'].ffill().bfill()
            except:
                df_processed['value'] = df_processed['value'].interpolate(method='linear').ffill().bfill()
        
        if resample_rate != 'Tiada':
            df_processed.set_index('timestamp', inplace=True)
            df_processed = df_processed.resample(resample_rate).mean()
            df_processed.reset_index(inplace=True)
            
        return df_processed, outliers_count, total_missing

    def get_audit_log(self):
        return self.log_stream.getvalue()

# --- Fungsi Bantuan untuk Paparan Pagination ---
def render_paginated_dataframe(df, key_prefix):
    col1, col2 = st.columns([1, 3])
    with col1:
        row_opts = ["100", "500", "Semua"]
        selected_rows = st.selectbox("Baris per halaman:", row_opts, key=f"{key_prefix}_rows")
    
    if selected_rows == "Semua":
        st.dataframe(df, use_container_width=True, height=400)
    else:
        rows_per_page = int(selected_rows)
        total_pages = max(1, len(df) // rows_per_page + (1 if len(df) % rows_per_page > 0 else 0))
        
        if f"{key_prefix}_page" not in st.session_state:
            st.session_state[f"{key_prefix}_page"] = 1
            
        # Kawalan navigasi
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Sebelumnya", key=f"{key_prefix}_prev", use_container_width=True):
                st.session_state[f"{key_prefix}_page"] = max(1, st.session_state[f"{key_prefix}_page"] - 1)
        with nav_col2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Halaman <b>{st.session_state[f'{key_prefix}_page']}</b> dari <b>{total_pages}</b></div>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Seterusnya ➡️", key=f"{key_prefix}_next", use_container_width=True):
                st.session_state[f"{key_prefix}_page"] = min(total_pages, st.session_state[f"{key_prefix}_page"] + 1)
        
        # Pengiraan indeks data
        start_idx = (st.session_state[f"{key_prefix}_page"] - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        
        # Paparan jadual
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)


def main():
    st.title("🌊 HydroData Processor Pro")
    processor = HydroDataProcessor()

    with st.sidebar:
        st.header("⚙️ Panel Kawalan")
        uploaded_file = st.file_uploader("1. Muat Naik Fail Data", type=['001', 'txt', 'csv', 'kud'])
        
        st.markdown("---")
        st.subheader("2. Parameter Penapisan (Filtering)")
        threshold = st.number_input("Threshold Nilai Oulier (±)", value=9.0, step=0.5)
        interpolation_method = st.selectbox("Kaedah Ganti Data", options=['linear', 'pad', 'polynomial', 'nearest'])
        
        st.markdown("---")
        st.subheader("3. Tetapan Sela Masa (Resampling)")
        resample_dict = {"Kekalkan Asal": "Tiada", "50 Saat (50S)": "50S", "1 Jam (1H)": "1H"}
        resample_choice = st.selectbox("Format Masa Output", options=list(resample_dict.keys()))
        resample_val = resample_dict[resample_choice]

        st.markdown("---")
        process_btn = st.button("🚀 PROSES DATA", use_container_width=True, type="primary")

    if uploaded_file is not None:
        # Pengekstrakan Fail
        if 'raw_df' not in st.session_state or st.session_state.uploaded_filename != uploaded_file.name:
            st.session_state.raw_df = processor.parse_file(uploaded_file)
            st.session_state.uploaded_filename = uploaded_file.name
            # Reset page states apabila fail baru dimuat naik
            st.session_state['raw_page'] = 1
            st.session_state['clean_page'] = 1
        
        raw_df = st.session_state.raw_df
        
        if raw_df is not None:
            tab1, tab2, tab3 = st.tabs(["📊 Paparan Data", "📈 Visualisasi Interaktif", "📝 Eksport & Log"])
            
            with tab1:
                st.metric("Jumlah Keseluruhan Data Mentah", f"{len(raw_df):,} baris")
                render_paginated_dataframe(raw_df, "raw")

            # --- ENJIN PEMPROSESAN DENGAN PROGRESS BAR ---
            if process_btn:
                # 1. Tunjuk Progress Bar UI
                progress_text = "Memulakan pemprosesan enjin data..."
                progress_bar = st.progress(0, text=progress_text)
                
                # Simulasi sedikit masa untuk UI/UX jika fail terlalu cepat diproses
                for percent_complete in range(0, 50, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete, text="Menapis outlier dan kelompongan data (Missing Data)...")
                
                # 2. Proses data sebenar
                clean_df, out_count, missing_count = processor.process_data(
                    raw_df, threshold=threshold, method=interpolation_method, resample_rate=resample_val)
                st.session_state.clean_df = clean_df
                st.session_state.out_count = out_count
                st.session_state.missing_count = missing_count
                
                # 3. Kemaskini Progress Bar ke 100%
                for percent_complete in range(50, 101, 10):
                    time.sleep(0.1)
                    progress_bar.progress(percent_complete, text="Menyusun semula sela masa (Resampling)...")
                
                time.sleep(0.5)
                progress_bar.empty() # Buang bar selepas siap
                
                # 4. Notifikasi Toast & Success
                st.toast('✅ Pemprosesan Selesai!', icon='🎉')
                st.success(f"Berjaya memproses {len(clean_df):,} baris data baharu!")
                    
            # --- PAPARAN SELEPAS DIPROSES ---
            if 'clean_df' in st.session_state:
                clean_df = st.session_state.clean_df
                
                with tab2:
                    st.subheader("Perbandingan Graf")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=raw_df['timestamp'], y=raw_df['value'], mode='lines', name='Asal (Raw)', line=dict(color='red', width=1, dash='dot')))
                    fig.add_trace(go.Scatter(x=clean_df['timestamp'], y=clean_df['value'], mode='lines', name='Diproses', line=dict(color='blue', width=2)))
                    fig.update_layout(title="Kualiti Data Ara Air", hovermode="x unified", height=500)
                    st.plotly_chart(fig, use_container_width=True)

                with tab3:
                    st.metric("Jumlah Data Baru Dieksport", f"{len(clean_df):,} baris")
                    render_paginated_dataframe(clean_df, "clean")
                    
                    st.markdown("---")
                    csv = clean_df.to_csv(index=False).encode('utf-8')
                    st.download_button(label="⬇️ Muat Turun Data Bersih (.CSV)", data=csv, file_name=f"Processed_{uploaded_file.name}.csv", mime="text/csv", use_container_width=True)

if __name__ == "__main__":
    main()
