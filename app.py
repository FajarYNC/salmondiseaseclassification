import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import requests

# Fungsi untuk mengunduh model dari Google Drive
def download_model(url, output_path):
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)

class SalmonDiseaseApp:
    def __init__(self, model_path):
        # Load model yang sudah dilatih
        self.model = tf.keras.models.load_model(model_path)
        self.input_shape = (224, 224, 3)
    
    def preprocess_image(self, image_path):
        # Baca gambar
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize gambar
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        
        # Normalisasi
        img = img / 255.0
        
        # Tambah dimensi batch
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path):
        # Preprocessing gambar
        processed_img = self.preprocess_image(image_path)
        
        # Prediksi
        prediction = self.model.predict(processed_img)[0][0]
        
        # Interpretasi hasil
        if prediction > 0.5:
            status = 'Infected'
            confidence = prediction * 100
        else:
            status = 'Fresh'
            confidence = (1 - prediction) * 100
        
        return status, confidence

def main():
    # Konfigurasi halaman
    st.set_page_config(
        page_title='Salmon Disease Classifier', 
        page_icon='🐟', 
        layout='wide'
    )

    # Sidebar dengan desain dashboard
    with st.sidebar:
        # Logo atau judul utama
        st.markdown("""
        <div style='text-align: center; margin-bottom: 20px;'>
            <h1 style='color: #2C3E50; font-size: 2rem;'>
                🐟 Salmon Health <br>Classifier
            </h1>
        </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown("---")

        # Informasi Aplikasi
        st.markdown("""
        ## 📊 Tentang Aplikasi
        Aplikasi ini menggunakan **Jaringan Saraf Tiruan Konvolusional (CNN)** untuk mengklasifikasikan kondisi kesehatan ikan salmon 
        melalui analisis citra digital.
        """)

        # Fitur Utama
        st.markdown("""
        ## 🔍 Fitur Utama
        - Klasifikasi penyakit ikan salmon
        - Analisis berbasis Deep Learning
        - Tingkat akurasi tinggi 
        """)

        # Panduan Penggunaan
        st.markdown("""
        ## 📝 Cara Pakai
        1. Unggah gambar ikan salmon
        2. Klik 'Analisis Penyakit'
        3. Lihat hasil klasifikasi~
        """)

        # Informasi Kontak/Kredit
        st.markdown("---")
        st.markdown("""
        ### 🎓 Proyek Pengolahan Citra
        Dibuat untuk memenuhi tugas mata kuliah 
        Pengolahan Citra
        """)

    # Konten Utama
    st.title('Salmon Disease Classification 🐟')
    
    # Model path
    model_path = 'salmon_disease_model.keras'
    
    # Unduh model dari Google Drive jika belum ada
    if not os.path.exists(model_path):
        with st.spinner('Mengunduh model...'):
            download_url = 'https://drive.google.com/uc?export=download&id=1vFRO7dr3rSpEp0MlyJtguoCfrvXsOUGO'
            download_model(download_url, model_path)
    
    # Verifikasi apakah file sudah ada
    if os.path.exists(model_path):
        try:
            app = SalmonDiseaseApp(model_path)
        except Exception as e:
            st.error(f'Error memuat model: {e}')
            st.stop()
    else:
        st.error('Gagal mengunduh model. Silakan periksa URL unduhan.')

    # Kolom untuk upload dan preview
    col1, col2 = st.columns([2, 1])

    with col1:
        # Upload gambar
        uploaded_file = st.file_uploader(
            'Pilih Gambar Ikan Salmon', 
            type=['jpg', 'jpeg', 'png']
        )
    
    if uploaded_file is not None:
        # Simpan file sementara
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        with col2:
            # Tampilkan gambar
            st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True)
        
        # Tombol prediksi
        if st.button('Analisis Penyakit'):
            try:
                # Lakukan prediksi
                status, confidence = app.predict(temp_path)
                
                # Tampilkan hasil
                if status == 'Infected':
                    st.error(f'🚨 Ikan Salmon Terinfeksi (Confidence: {confidence:.2f}%)')
                else:
                    st.success(f'✅ Ikan Salmon Sehat (Confidence: {confidence:.2f}%)')
                
                # Hapus file sementara
                os.remove(temp_path)
            
            except Exception as e:
                st.error(f'Gagal melakukan prediksi: {e}')

if __name__ == '__main__':
    main()
