import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

MODEL_PATH = "model.h5"
GDRIVE_URL = "https://drive.google.com/uc?id=/1pV6sy94M1-6_Q4fVGUyhprX_YHsROckt"  # 👈 여기에 실제 파일 ID 입력

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 모델을 Google Drive에서 다운로드 중..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("🩺 Chest X-ray Pneumonia Detection")
st.write(
    """
    이 앱은 흉부 X-ray 이미지를 분석하여 폐렴 여부를 예측합니다.  
    아래에서 이미지를 업로드하면 결과가 바로 출력됩니다.
    """
)

uploaded_file = st.file_uploader("이미지를 업로드하세요 (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    if st.button("🔍 예측하기"):
        with st.spinner("이미지 분석 중..."):
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]
            label = "Pneumonia (폐렴)" if prediction > 0.5 else "Normal (정상)"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.markdown(f"### ✅ 예측 결과: **{label}**")
            st.write(f"신뢰도: {confidence:.2%}")
