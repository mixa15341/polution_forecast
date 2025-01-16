import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pickle
from fastapi.middleware.cors import CORSMiddleware
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены (для тестирования)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы
    allow_headers=["*"],  # Разрешить все заголовки
)

# Загрузка модели и scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Функция для создания временного ряда
def create_dataset(data, look_back=5):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        Y.append(data[i + look_back])
    return np.array(X), np.array(Y)

# Эндпоинт для прогнозирования
@app.post("/predict/")
async def predict(station_code: str, target_variable: str, future_steps: int = 30):
    try:
        logger.info(f"Получен запрос: station_code={station_code}, target_variable={target_variable}, future_steps={future_steps}")
        
        # Загрузка данных
        df = pd.read_csv('polution_data.csv')
        logger.info(f"Загружено {len(df)} строк данных.")
        
        # Приведение Station code к строке
        df['Station code'] = df['Station code'].astype(str)
        station_code = str(station_code)

        # Фильтрация данных
        filtered_df = df[df['Station code'] == station_code]
        logger.info(f"Найдено {len(filtered_df)} строк для станции {station_code}.")

        # Проверка, что данные не пустые
        if filtered_df.empty:
            logger.error(f"Данные для станции {station_code} не найдены.")
            raise HTTPException(status_code=404, detail=f"Данные для станции {station_code} не найдены.")
        
        # Преобразование в временной ряд
        pivot = filtered_df.pivot(index='Measurement date', columns='Station code', values=target_variable).fillna(0)
        data = pivot.values
        logger.info(f"Данные для переменной {target_variable}: {data}")

         # Проверка, что данные не пустые после преобразования
        if data.size == 0:
            logger.error(f"Данные для переменной {target_variable} не найдены.")
            raise HTTPException(status_code=404, detail=f"Данные для переменной {target_variable} не найдены.")

        # Нормализация данных
        data_scaled = scaler.transform(data)

        # Используем последние look_back значений для предсказания
        look_back = 5
        last_sequence = data_scaled[-look_back:]

        # Прогнозирование на future_steps шагов вперед
        future_predictions = []
        for _ in range(future_steps):
            next_prediction = model.predict(last_sequence[np.newaxis, :, :])
            future_predictions.append(next_prediction[0])
            last_sequence = np.vstack([last_sequence[1:], next_prediction])

        # Возвращение данных в исходный масштаб
        future_predictions = scaler.inverse_transform(future_predictions)

        # Создание дат для будущих предсказаний
        last_date = pd.to_datetime(pivot.index[-1])
        future_dates = [last_date + timedelta(days=i) for i in range(1, future_steps + 1)]

        # Формирование результата
        result = {
            "future_dates": [str(date) for date in future_dates],
            "predictions": [float(value) for value in future_predictions.flatten()]
        }

        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
