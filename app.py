import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Заголовок приложения
st.title("Прогнозирование загрязнения воздуха")

# Загрузка данных о станциях
try:
    df = pd.read_csv('polution_data.csv')
    logger.info(f"Загружено {len(df)} строк данных.")
except Exception as e:
    logger.error(f"Ошибка при загрузке данных: {e}")
    st.error(f"Ошибка при загрузке данных: {e}")

station_codes = df['Station code'].unique()
target_options = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']

# Выбор станции и целевой переменной
station_code = st.selectbox("Выберите станцию:", station_codes)
target_variable = st.selectbox("Выберите целевую переменную:", target_options)
future_steps = st.slider("Количество дней для прогноза:", 1, 60, 30)

# Кнопка для запуска прогнозирования
if st.button("Прогнозировать"):
    try:
        # Логирование данных перед отправкой
        logger.info(f"Данные для отправки: station_code={station_code}, target_variable={target_variable}, future_steps={future_steps}")
        st.write("Данные для отправки:", {
            "station_code": station_code,
            "target_variable": target_variable,
            "future_steps": future_steps
        })

        # Отправка запроса с query-параметрами
        response = requests.post(
            "https://polution-forecast-1.onrender.com/predict/",
            params={
                "station_code": station_code,
                "target_variable": target_variable,
                "future_steps": future_steps
            }
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Результат: {result}")
        st.write("Результат:", result)

        # Обработка результата
        future_dates = result["future_dates"]
        predictions = result["predictions"]

        # Создание DataFrame для визуализации
        df_predictions = pd.DataFrame({
            "Дата": future_dates,
            "Прогноз": predictions
        })

        # Отображение таблицы
        st.write("Результаты прогнозирования:")
        st.dataframe(df_predictions)

        # Визуализация графика
        plt.figure(figsize=(10, 6))
        plt.plot(df_predictions["Дата"], df_predictions["Прогноз"], marker='o')
        plt.title(f"Прогноз {target_variable} для станции {station_code}")
        plt.xlabel("Дата")
        plt.ylabel("Концентрация")
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Отображение на карте
        st.write("Расположение станции на карте:")
        
        # Получение координат выбранной станции
        station_data = df[df['Station code'] == station_code].iloc[0]
        latitude, longitude = station_data['Latitude'], station_data['Longitude']

        # Создание карты
        m = folium.Map(location=[latitude, longitude], zoom_start=12)

        # Добавление маркера для станции
        folium.Marker(
            location=[latitude, longitude],
            popup=f"Станция: {station_code}\nПрогноз: {predictions[-1]:.2f}"
        ).add_to(m)

        # Отображение карты в Streamlit
        folium_static(m)

    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при отправке запроса: {e}")
        logger.error(f"Ответ сервера: {response.text}")  # Вывод текста ответа в терминал
        st.error(f"Ошибка при отправке запроса: {e}")
        st.write("Ответ сервера:", response.text)  # Вывод текста ответа в интерфейс
    except ValueError as e:
        logger.error(f"Ошибка при обработке ответа: {e}")
        st.error(f"Ошибка при обработке ответа: {e}")
