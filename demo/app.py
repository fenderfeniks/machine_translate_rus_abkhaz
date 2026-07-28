import os

import requests
import streamlit as st


API_BASE_URL = os.getenv("API_URL", "http://localhost:8000/api/v1")
API_KEY = os.getenv("API_KEY", "")

GENERATE_URL = f"{API_BASE_URL}/generate"
STREAM_URL = f"{API_BASE_URL}/generate/stream"

st.set_page_config(page_title="LLM Demo", page_icon="🤖", layout="centered")
st.title("🤖 LLM Demo")
st.caption("Генерация текста через REST API")

# ── Боковая панель ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Настройки")
    use_stream = st.toggle("Стриминг (SSE)", value=True)
    st.divider()
    st.markdown("**Лимит:** 5 запросов/мин  \n**Макс. длина промпта:** 2000 символов")

# ── Основной интерфейс ────────────────────────────────────────────────────────
prompt = st.text_area(
    "Промпт:",
    height=180,
    max_chars=2000,
    placeholder="Объясни, что такое Retrieval-Augmented Generation (RAG).",
)

headers = {"X-API-Key": API_KEY} if API_KEY else {}

if st.button("Сгенерировать", type="primary", disabled=not prompt.strip()):
    payload = {"prompt": prompt.strip()}

    if use_stream:
        # ── Стриминговый режим (/generate/stream) ────────────────────────────
        with st.spinner("Генерирую..."):
            try:
                output_box = st.empty()
                accumulated = ""

                with requests.post(
                    STREAM_URL,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            accumulated += chunk
                            output_box.markdown(accumulated)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    st.warning("⏳ Превышен лимит запросов. Подождите минуту и попробуйте снова.")
                else:
                    st.error(f"Ошибка сервера: {e}")
            except requests.exceptions.ConnectionError:
                st.error("Не удаётся подключиться к API. Проверьте, что сервер запущен.")
            except requests.exceptions.Timeout:
                st.error("Сервер не ответил за 60 секунд.")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка запроса: {e}")

    else:
        # ── Блокирующий режим (/generate) ────────────────────────────────────
        with st.spinner("Генерирую..."):
            try:
                resp = requests.post(
                    GENERATE_URL,
                    json=payload,
                    headers=headers,
                    timeout=120,
                )
                resp.raise_for_status()

                result = resp.json()
                generated_text = result.get("generated_text", "")
                st.markdown(generated_text)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    st.warning("⏳ Превышен лимит запросов. Подождите минуту и попробуйте снова.")
                else:
                    st.error(f"Ошибка сервера: {e}")
            except requests.exceptions.ConnectionError:
                st.error("Не удаётся подключиться к API. Проверьте, что сервер запущен.")
            except requests.exceptions.Timeout:
                st.error("Сервер не ответил за 120 секунд.")
            except requests.exceptions.RequestException as e:
                st.error(f"Ошибка запроса: {e}")

elif not prompt.strip():
    st.caption("Введите промпт, чтобы активировать кнопку.")
