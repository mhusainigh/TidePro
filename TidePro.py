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
            handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(handler)
            
    def parse_file(self, uploaded_file):
        filename = uploaded_file.name.lower()
        self.logger.info(f"Membaca fail: {uploaded_file.name}")
        try:
            content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            
            if "<VT t=" in content:
                content = content.replace('""', '"')
                data = re.findall(r'<VT t="([^"]+)">\s*([0-9.-]+)\s*</VT>', content)
                df = pd.DataFrame(data, columns=['timestamp', 'value'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df
                
            elif "---" in content[:1000] or ".csv" in filename:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['date', 'time', 'value'], na_values=['---'])
                if 'date' in df.columns and 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format="%Y.%m.%d %H:%M:%S", errors='coerce')
                    df = df[['timestamp', 'value']]
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                return df.dropna(subset=['timestamp'])
                
            elif ".001" in filename:
                lines = content.splitlines()
                start_time, raw_values, start_reading = None, [], False
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
                            raw_values.append(np.nan if val == -999 else val / 100.0)
                if start_time and raw_values:
                    timestamps = pd.date_range(start=start_time, periods=len(raw_values), freq='10S')
                    return pd.DataFrame({'timestamp': timestamps, 'value': raw_values})
                    
            elif ".kud" in filename:
                lines = content.splitlines()
                raw_values = [float(line.strip()) / 100.0 for line in lines[50:] if re.match(r'^-?\d+$', line.strip())]
                timestamps = pd.date_range(start=datetime.now().replace(hour=0, minute=0, second=0), periods=len(raw_values), freq='50S')
                return pd.DataFrame({'timestamp': timestamps, 'value': raw_values})
            else:
                self.logger.error(f"Format tidak disokong: {filename}")
                return None
        except Exception as e:
            self.logger.error(f"Ralat Pembacaan {filename}: {str(e)}")
            return None

    def process_data(self, df, threshold=0.09, out_method='PCHIP', miss_method='Cubic Spline', resample_rate='50S'):
        df_processed = df.copy()
        
        diffs = df_processed['value'].diff().abs()
        outliers_mask = diffs > threshold
        outliers_count = outliers_mask.sum()
        df_processed.loc[outliers_mask, 'value'] = np.nan
        self.logger.info(f"Dikesan {outliers_count} outlier melebihi threshold {threshold}m.")

        total_missing = df_processed['value'].isna().sum()
        if total_missing > 0:
            try:
                method_map = {
                    'Linear': 'linear', 'Cubic Spline': 'spline', 'PCHIP': 'pchip',
                    'Moving Average': 'linear', 'Harmonic': 'pchip',
                    'Kalman Filter': 'pchip', 'GPR': 'spline', 'LSTM': 'pchip'
                }
                pd_method = method_map.get(miss_method, 'linear')

                if miss_method == 'Moving Average':
                    df_processed['value'] = df_processed['value'].fillna(df_processed['value'].rolling(window=5, min_periods=1, center=True).mean())
                elif pd_method == 'spline':
                    df_processed['value'] = df_processed['value'].interpolate(method='spline', order=3)
                else:
                    df_processed['value'] = df_processed['value'].interpolate(method=pd_method)
                
                df_processed['value'] = df_processed['value'].ffill().bfill()
                self.logger.info(f"Berjaya interpolasi {total_missing} titik menggunakan proksi {miss_method}.")
            except Exception as e:
                self.logger.warning(f"Ralat interpolasi {miss_method}: {e}. Menukar ke Linear.")
                df_processed['value'] = df_processed['value'].interpolate(method='linear').ffill().bfill()
        
        if resample_rate != 'Tiada':
            df_processed.set_index('timestamp', inplace=True)
            df_processed = df_processed.resample(resample_rate).mean()
            df_processed.reset_index(inplace=True)
            self.logger.info(f"Resampling masa ke {resample_rate} selesai.")
            
        return df_processed, outliers_count, total_missing

    def format_output(self, df, structure_style):
        # Pengurusan format output berdasarkan pilihan nama struktur baru
        if "Structure 1" in structure_style:
            df_out = df.copy()
            df_out['time_str'] = df_out['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
            return df_out[['time_str', 'value']]
        elif "Structure 2" in structure_style:
            df_out = df.copy()
            df_out['date_str'] = df_out['timestamp'].dt.strftime('%Y-%m-%d')
            df_out['time_str'] = df_out['timestamp'].dt.strftime('%H:%M')
            return df_out[['date_str', 'time_str', 'value']]
        else:
            # Mengembalikan format DataFrame lalai untuk Structure 3-7 buat sementara waktu
            return df

    def get_audit_log(self):
        return self.log_stream.getvalue()

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
            
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
        with nav_col1:
            if st.button("⬅️ Sebelumnya", key=f"{key_prefix}_prev", use_container_width=True):
                st.session_state[f"{key_prefix}_page"] = max(1, st.session_state[f"{key_prefix}_page"] - 1)
        with nav_col2:
            st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Halaman <b>{st.session_state[f'{key_prefix}_page']}</b> dari <b>{total_pages}</b></div>", unsafe_allow_html=True)
        with nav_col3:
            if st.button("Seterusnya ➡️", key=f"{key_prefix}_next", use_container_width=True):
                st.session_state[f"{key_prefix}_page"] = min(total_pages, st.session_state[f"{key_prefix}_page"] + 1)
        
        start_idx = (st.session_state[f"{key_prefix}_page"] - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True, height=400)


def main():
    st.title("🌊 HydroData Processor Pro")
    processor = HydroDataProcessor()

    with st.sidebar:
        st.header("⚙️ Panel Kawalan")
        uploaded_files = st.file_uploader("1. Muat Naik Fail Data", type=['001', 'txt', 'csv', 'kud'], accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("2. Parameter Penapisan")
        threshold = st.number_input("Threshold Kadar Perubahan (m) cth 0.09", value=0.09, step=0.01)
        outlier_method = st.selectbox("Kaedah Ganti Outlier", options=['PCHIP', 'Harmonic', 'Moving Average', 'Kalman Filter', 'GPR', 'LSTM'])
        missing_method = st.selectbox("Kaedah Missing Data", options=['Linear', 'Cubic Spline', 'PCHIP', 'Harmonic', 'Kalman Filter', 'GPR', 'LSTM'])
        
        st.markdown("---")
        st.subheader("3. Tetapan Output Final")
        resample_dict = {"Kekalkan Asal": "Tiada", "50 Saat (50S)": "50S", "1 Jam (1H)": "1H"}
        resample_val = resample_dict[st.selectbox("Format Masa Output", options=list(resample_dict.keys()))]
        
        # UI BARU: Radio Button dengan Captions untuk Struktur Data
        st.write("**Gaya Struktur Data:**")
        structure_options = [
            "Structure 1: Datetime-Value",
            "Structure 2: Date-Time-Value",
            "Structure 3: PSMSL Annual",
            "Structure 4: Decimal Year",
            "Structure 5: Block Daily",
            "Structure 6: Sequential Numeric",
            "Structure 7: Instrument Raw"
        ]
        structure_captions = [
            "Contoh: 14/2/2025 0:00, 1.2",
            "Contoh: 2024-02-01, 00:00, 2.34",
            "Contoh: 1986;  7015;N;000",
            "Contoh: 1984.9583;  7081; 0;000",
            "Blok data harian dengan bacaan ke bawah",
            "Matriks/Susunan nombor berurutan",
            "Format asal (Raw) bacaan alat"
        ]
        structure_style = st.radio(
            label="Sila pilih struktur pangkalan data:", 
            options=structure_options, 
            captions=structure_captions,
            label_visibility="collapsed"
        )
        
        # UI BARU: Dropdown Lengkap untuk Ekstensi Fail & Stesen
        st.write("")
        ext_options = [
            '.csv', '.txt', '.dat', '.psmsl', 
            '.LAN', '.PEN', '.LUM', '.PTK', '.TGK', '.KUK', '.JBH', 
            '.SED', '.TIO', '.NKP', '.CHD', '.GET', '.LAK', '.KCH', 
            '.BTU', '.MYY', '.KKB', '.KUD', '.SDK', '.LDU', '.TWU', '.LBU'
        ]
        ext_choice = st.selectbox("Format Ekstensi Fail (Output):", options=ext_options)

        st.markdown("---")
        process_btn = st.button("🚀 PROSES DATA", use_container_width=True, type="primary")

    if uploaded_files:
        if 'raw_data_dict' not in st.session_state or st.session_state.get('uploaded_count') != len(uploaded_files):
            st.session_state.raw_data_dict = {}
            st.session_state.file_intervals = {}
            for f in uploaded_files:
                df = processor.parse_file(f)
                if df is not None:
                    st.session_state.raw_data_dict[f.name] = df
                    if len(df) > 1:
                        interval = df['timestamp'].diff().mode()[0]
                        st.session_state.file_intervals[f.name] = interval
            st.session_state.uploaded_count = len(uploaded_files)

        unique_intervals = set(st.session_state.file_intervals.values())
        is_uniform = len(unique_intervals) <= 1

        if not is_uniform:
            st.error("🛑 RALAT: Pemprosesan Dihentikan! Fail-fail yang dimuat naik mempunyai sela masa asal yang berbeza.")
            st.markdown("**Sila pastikan semua fail dalam satu sesi muat naik mempunyai sela masa yang sama.**")
            for fname, inv in st.session_state.file_intervals.items():
                time_str = str(inv).split()[-1] if pd.notna(inv) else "Tidak Diketahui"
                st.write(f"- 📄 `{fname}` : **{time_str}** (Jam:Minit:Saat)")

        if process_btn:
            if not is_uniform:
                st.toast("Ralat: Sela masa berbeza. Sila buang fail yang tidak seragam.", icon="❌")
            else:
                st.session_state.processed_dict = {}
                st.session_state.audit_log = ""
                progress_bar = st.progress(0, text="Memproses data...")
                
                for i, f in enumerate(uploaded_files):
                    raw_df = st.session_state.raw_data_dict.get(f.name)
                    if raw_df is not None:
                        clean_df, out_cnt, miss_cnt = processor.process_data(
                            raw_df, threshold, outlier_method, miss_method=missing_method, resample_rate=resample_val
                        )
                        final_df = processor.format_output(clean_df, structure_style)
                        st.session_state.processed_dict[f.name] = {
                            'clean': clean_df, 'final': final_df, 'out_cnt': out_cnt, 'miss_cnt': miss_cnt
                        }
                    progress_bar.progress(int(((i+1)/len(uploaded_files))*100))
                
                st.session_state.audit_log = processor.get_audit_log()
                time.sleep(0.5)
                progress_bar.empty()
                st.toast('✅ Pemprosesan Selesai!', icon='🎉')

        # --- UI PAPARAN TERPERINCI ---
        st.markdown("### 🔍 Pemapar Data Terperinci")
        selected_file = st.selectbox("Pilih fail untuk dipaparkan:", [f.name for f in uploaded_files])
        
        if selected_file:
            raw_df = st.session_state.raw_data_dict.get(selected_file)
            processed_data = st.session_state.get('processed_dict', {}).get(selected_file)
            
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Asal", "📈 Graf Perbandingan", "📝 Data Diproses", "📋 Log & Eksport"])
            
            with tab1:
                if raw_df is not None:
                    st.metric("Jumlah Baris Asal", f"{len(raw_df):,}")
                    render_paginated_dataframe(raw_df, f"raw_{selected_file}")
            
            with tab2:
                if processed_data:
                    clean_df = processed_data['clean']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=raw_df['timestamp'], y=raw_df['value'], mode='markers+lines', name='Asal (Raw)', line=dict(color='red', width=1, dash='dot'), marker=dict(size=3)))
                    fig.add_trace(go.Scatter(x=clean_df['timestamp'], y=clean_df['value'], mode='lines', name='Diproses (Bersih)', line=dict(color='blue', width=2)))
                    fig.update_layout(title=f"Analisis Outlier: {selected_file}", hovermode="x unified", height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Sila tekan butang 'PROSES DATA' untuk melihat graf perbandingan.")

            with tab3:
                if processed_data:
                    st.write(f"**Outlier dibuang:** {processed_data['out_cnt']} | **Titik dibaiki:** {processed_data['miss_cnt']}")
                    render_paginated_dataframe(processed_data['final'], f"clean_{selected_file}")
                else:
                    st.info("Sila tekan butang 'PROSES DATA' dahulu.")

            with tab4:
                if processed_data:
                    st.text_area("Audit Trail (Log Enjin)", value=st.session_state.get('audit_log', ''), height=200)
                    
                    # Logik Eksport
                    is_csv = ext_choice == '.csv'
                    csv_data = processed_data['final'].to_csv(index=False, header=is_csv).encode('utf-8')
                    new_filename = f"{selected_file.split('.')[0]}_processed{ext_choice}"
                    st.download_button(label=f"⬇️ Muat Turun Data ({ext_choice})", data=csv_data, file_name=new_filename, mime="text/plain", use_container_width=True)

if __name__ == "__main__":
    main()
