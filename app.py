import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import joblib
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode, RTCConfiguration
import logging

# Inisialisasi logging
logging.basicConfig(level=logging.INFO)

# Simpan kredensial pengguna di session_state (untuk demo; gunakan database nyata dalam implementasi sebenarnya)
if "users" not in st.session_state:
    st.session_state["users"] = {
        "admin@example.com": {"username": "admin", "password": "password123"}
    }

# Load model Keras
try:
    model = load_model('model.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logging.info("Model berhasil dimuat")
except Exception as e:
    st.error(f"Error loading model: {e}")
    logging.error(f"Error loading model: {e}")

# Load model SVM
try:
    svm_model = joblib.load('svm_model.pkl')
    logging.info("SVM model berhasil dimuat")
except Exception as e:
    st.error(f"Error loading SVM model: {e}")
    logging.error(f"Error loading SVM model: {e}")

# Fungsi untuk pra-pemrosesan gambar/frame dan membuat prediksi
def preprocess_image(image):
    try:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = image.resize((300, 300))  # Sesuaikan target_size jika perlu
        image_array = np.array(image) / 255.0  # Normalisasi jika perlu
        if image_array.shape[-1] == 4:
            image_array = image_array[..., :3]  # Konversi RGBA ke RGB jika perlu
        image_array = np.expand_dims(image_array, axis=0)  # Tambahkan dimensi batch
        logging.info(f"Image preprocessed: {image_array.shape}")
        return image_array
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        st.error(f"Error preprocessing image: {e}")
        return None

def predict(image):
    try:
        processed_image = preprocess_image(image)
        if processed_image is not None:
            cnn_features = model.predict(processed_image)
            svm_prediction = svm_model.predict(cnn_features)
            result = 'Kaleng Cacat' if svm_prediction[0] == 0 else 'Kaleng Tidak Cacat'
            logging.info(f"Prediction made: {result}")
            return result
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        st.error(f"Error during prediction: {e}")
        return None

def detect_can(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust contour area as necessary
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 1.5:  # Adjust aspect ratio range as necessary
                return True, (x, y, w, h)
    return False, None

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.frame_counter = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        logging.info("Frame diterima untuk transformasi")

        # Tambahkan penghitung frame
        self.frame_counter += 1
        logging.info(f"Memproses frame {self.frame_counter}")

        # Periksa apakah frame mengandung objek seperti kaleng
        has_can, bbox = detect_can(img)
        if has_can:
            x, y, w, h = bbox
            can_roi = img[y:y+h, x:x+w]
            result = predict(can_roi)
            if result:
                cv2.putText(img, result, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                logging.info(f"Hasil klasifikasi: {result}")

        return img

# Fungsi untuk halaman login
def login():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        users = st.session_state["users"]
        if email in users and users[email]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = users[email]["username"]
            logging.info(f"User {email} berhasil login")
        else:
            st.error("Email atau password salah")
            logging.warning(f"Gagal login untuk email: {email}")

# Fungsi untuk halaman register
def register():
    st.title("Register")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    if st.button("Register"):
        users = st.session_state["users"]
        if email in users:
            st.error("Email sudah terdaftar")
            logging.warning(f"Pendaftaran dengan email yang sudah terdaftar: {email}")
        else:
            users[email] = {"username": username, "password": password}
            st.session_state["users"] = users
            st.success("Pendaftaran berhasil. Silakan login.")
            logging.info(f"Pengguna baru terdaftar dengan email: {email}")

# Fungsi untuk halaman klasifikasi
def app():
    st.title("Can Classifier")
    st.write(f"Selamat datang, {st.session_state['username']}!")
    st.write("Aplikasi ini mengklasifikasikan kaleng sebagai cacat atau tidak cacat.")

    mode = st.radio("Pilih mode:", ('Klasifikasi Real-Time', 'Unggah Gambar'))

    if mode == 'Klasifikasi Real-Time':
        rtc_configuration = RTCConfiguration({
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]}
            ]
        })

        webrtc_streamer(
            key="example",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=rtc_configuration,
        )
    elif mode == 'Unggah Gambar':
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Gambar yang diunggah.', use_column_width=True)
            st.write("")
            st.write("Mengklasifikasikan...")

            result = predict(np.array(image))

            st.write(f"Kaleng tersebut **{result}**.")

# Main loop
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if st.session_state["logged_in"]:
    app()
else:
    choice = st.selectbox("Login/Daftar", ["Login", "Register"])
    if choice == "Login":
        login()
    else:
        register()
