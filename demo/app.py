import os

import requests
import streamlit as st


# Используем актуальный эндпоинт для классификации
API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/classify")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="Детектор Фейковых Новостей", page_icon="🕵️‍♂️")
st.title("Детектор Фейковых Новостей")
st.markdown("Введите текст новости, чтобы проверить, является ли она достоверной.")

# Простое текстовое поле вместо чата
prompt = st.text_area("Текст новости:", height=200)

if st.button("Проверить"):
    if prompt.strip():
        # Отправляем схему ClassificationRequest
        payload = {"text": prompt}
        headers = {"X-API-Key": API_KEY} if API_KEY else {}

        with st.spinner("Анализирую текст..."):
            try:
                response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
                response.raise_for_status()

                result = response.json()
                label_id = result.get("label_id")
                confidence = result.get("confidence", 0) * 100

                # Предположим, что 1 - это фейк, а 0 - реальная новость
                # (можешь поменять местами в зависимости от твоего датасета)
                if label_id == 1:
                    st.error(f"🚨 Вероятнее всего, это ФЕЙК! (Уверенность: {confidence:.2f}%)")
                else:
                    st.success(f"✅ Похоже на правду. (Уверенность: {confidence:.2f}%)")

                with st.expander("Детали предсказания (Вероятности классов)"):
                    st.json(result)

            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка связи с сервером: {e}")
    else:
        st.warning("Пожалуйста, введите текст для проверки.")
