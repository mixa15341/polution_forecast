import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import folium
from streamlit_folium import folium_static



# Заголовок приложения
st.title("Прогнозирование загрязнения воздуха")

# Загрузка данных о станциях
df = pd.read_csv('polution_data.csv')
station_codes = df['Station code'].unique()
target_options = ['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']

# Выбор станции и целевой переменной
station_code = st.selectbox("Выберите станцию:", station_codes)
target_variable = st.selectbox("Выберите целевую переменную:", target_options)
future_steps = st.slider("Количество дней для прогноза:", 1, 60, 30)

# Кнопка для запуска прогнозирования
if st.button("Прогнозировать"):
    # Отправка запроса на сервер FastAPI
    try:
        response = requests.post(
            "https://polution-forecast-1.onrender.com/predict/",
            json={
                "station_code": str(station_code),
                "target_variable": target_variable,
                "future_steps": future_steps
            }
        )
        response.raise_for_status()

        # Получение и отображение результата
        result = response.json()
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
        st.error(f"Ошибка при отправке запроса: {e}")
    except ValueError as e:
        st.error(f"Ошибка при обработке ответа: {e}")
