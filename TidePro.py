import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import plotly.graph_objects as go
from datetime import datetime
import logging

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
        """Modul Membaca Fail (Mengekalkan fungsi sedia ada yang betul)"""
        filename = uploaded_file.name.lower()
        self.logger.info(f"--- Mula membaca fail: {uploaded_file.name} ---")
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
                self.logger.error("Format tidak disokong.")
                return None
        except Exception as e:
            self.logger.error(f"Ralat Pembacaan: {str(e)}")
            return None

    def process_data(self, df, threshold=0.09, method='pchip', resample_rate='50S'):
        """Modul Pemprosesan (Ditambahbaik dengan Rate of Change & PCHIP)"""
        df_processed = df.copy()
        
        # 1. OUTLIER DETECTION (Kadar Perubahan / Rate of Change)
        # Mencari beza antara titik semasa dan sebelumnya
        diffs = df_processed['value'].diff().abs()
        outliers_mask = diffs > threshold
        outliers_count = outliers_mask.sum()
        df_processed.loc[outliers_mask, 'value'] = np.nan
        self.logger.info(f"Dikesan {outliers_count} outlier (lonjakan > {threshold}m).")

        # 2. INTERPOLASI (Isi missing data)
        total_missing = df_processed['value'].isna().sum()
        if total_missing > 0:
            try:
                if method == 'cubic spline':
                    df_processed['value'] = df_processed['value'].interpolate(method='spline', order=3)
                elif method == 'pchip':
                    df_processed['value'] = df_processed['value'].interpolate(method='pchip')
                elif method == 'moving average':
                    df_processed['value'] = df_processed['value'].fillna(df_processed['value'].rolling(window=5, min_periods=1, center=True).mean())
                else: # linear, pad, etc
                    df_processed['value'] = df_processed['value'].interpolate(method=method)
                
                df_processed['value'] = df_processed['value'].ffill().bfill()
                self.logger.info(f"Berjaya membaiki {total_missing} data menggunakan kaedah {method.upper()}.")
            except Exception as e:
                self.logger.error(f"Gagal interpolasi {method}: {e}. Tukar ke Linear.")
                df_processed['value'] = df_processed['value'].interpolate(method='linear').ffill().bfill()
        
        # 3. RESAMPLING
        if resample_rate != 'Tiada':
            df_processed.set_index('timestamp', inplace=True)
            df_processed = df_processed.resample(resample_rate).mean()
            df_processed.reset_index(inplace=True)
            self.logger.info(f"Sela masa ditukar ke {resample_rate}.")
            
        return df_processed

    def format_output(self, df, structure_style):
        """Memformat data mengikut pilihan struktur pengguna"""
        if structure_style == "Struktur-1 (DD/MM/YYYY H:MM, Nilai)":
            df_out = df.copy()
            df_out['time_str'] = df_out['timestamp'].dt.strftime('%d/%m/%Y %H:%M')
            return df_out[['time_str', 'value']]
        elif structure_style == "Struktur-2 (YYYY-MM-DD, HH:MM, Nilai)":
            df_out = df.copy()
            df_out['date_str'] = df_out['timestamp'].dt.strftime('%Y-%m-%d')
            df_out['time_str'] = df_out['timestamp'].dt.strftime('%H:%M')
            return df_out[['date_str', 'time_str', 'value']]
        else:
            return df

    def get_audit_log(self):
        return self.log_stream.getvalue()

def main():
    st.title("🌊 HydroData Processor Pro")
    
    # Simpan log ke dalam session state supaya tidak hilang
    if 'audit_log' not in st.session_state:
        st.session_state.audit_log = ""
    
    processor = HydroDataProcessor()

    with st.sidebar:
        st.header("⚙️ Panel Kawalan")
        # Fungsi BERBILANG FAIL (Multiple Files)
        uploaded_files = st.file_uploader("1. Muat Naik Fail Data", type=['001', 'txt', 'csv', 'kud'], accept_multiple_files=True)
        
        st.markdown("---")
        st.subheader("2. Parameter Penapisan")
        # Nilai lalai ditukar kepada 0.09m (9cm)
        threshold = st.number_input("Threshold Kadar Perubahan (m) cth 0.09 untuk 9cm", value=0.09, step=0.01)
        interpolation_method = st.selectbox("Kaedah Penyelesaian", options=['pchip', 'cubic spline', 'linear', 'pad', 'moving average'])
        
        st.markdown("---")
        st.subheader("3. Tetapan Output")
        resample_dict = {"Kekalkan Asal": "Tiada", "50 Saat (50S)": "50S", "1 Jam (1H)": "1H"}
        resample_val = resample_dict[st.selectbox("Format Masa Output", options=list(resample_dict.keys()))]
        
        structure_style = st.selectbox("Gaya Struktur Data (Final)", options=[
            "Struktur-1 (DD/MM/YYYY H:MM, Nilai)",
            "Struktur-2 (YYYY-MM-DD, HH:MM, Nilai)",
            "Lalai (Default CSV)"
        ])
        
        ext_choice = st.selectbox("Format Format Fail", options=['.csv', '.txt', '.dat', '.kud'])

        st.markdown("---")
        process_btn = st.button("🚀 PROSES SEMUA FAIL", use_container_width=True, type="primary")

    if uploaded_files:
        st.info(f"📁 {len(uploaded_files)} fail sedia untuk diproses.")
        
        if process_btn:
            st.session_state.processed_results = []
            st.session_state.audit_log = "" # Reset log baru
            
            progress_bar = st.progress(0, text="Memulakan Pemprosesan Kumpulan...")
            
            for i, file in enumerate(uploaded_files):
                # Update progress
                progress_bar.progress(int((i / len(uploaded_files)) * 100), text=f"Memproses: {file.name}")
                
                # Baca fail
                raw_df = processor.parse_file(file)
                if raw_df is not None:
                    # Proses fail
                    clean_df = processor.process_data(raw_df, threshold, interpolation_method, resample_val)
                    # Format output
                    final_df = processor.format_output(clean_df, structure_style)
                    
                    st.session_state.processed_results.append({
                        'name': file.name,
                        'clean_df': clean_df,  # Untuk graf
                        'final_df': final_df   # Untuk muat turun
                    })
            
            st.session_state.audit_log += processor.get_audit_log()
            progress_bar.progress(100, text="Selesai!")
            st.toast(f'✅ {len(uploaded_files)} fail siap diproses!', icon='🎉')
            
        # Paparan Hasil (Jika ada)
        if 'processed_results' in st.session_state and len(st.session_state.processed_results) > 0:
            tab1, tab2, tab3 = st.tabs(["📈 Analisis Visual", "📥 Muat Turun Data", "📝 Audit Trail"])
            
            with tab1:
                selected_result = st.selectbox("Pilih fail untuk lihat graf:", [res['name'] for res in st.session_state.processed_results])
                for res in st.session_state.processed_results:
                    if res['name'] == selected_result:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=res['clean_df']['timestamp'], y=res['clean_df']['value'], mode='lines', name='Data Bersih'))
                        fig.update_layout(title=f"Graf: {res['name']}", hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.write("Senarai Fail Siap Diproses:")
                for res in st.session_state.processed_results:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"📄 **{res['name']}**")
                    with col2:
                        csv = res['final_df'].to_csv(index=False, header=False).encode('utf-8')
                        new_filename = f"{res['name'].split('.')[0]}_processed{ext_choice}"
                        st.download_button(label="Muat Turun", data=csv, file_name=new_filename, mime="text/plain", key=f"dl_{res['name']}")

            with tab3:
                st.text_area("Log Pindaan (Audit Trail)", value=st.session_state.audit_log, height=400)
                st.download_button(label="Muat Turun Log", data=st.session_state.audit_log.encode('utf-8'), file_name="AuditTrail.txt")

if __name__ == "__main__":
    main()
