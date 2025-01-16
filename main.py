from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import pickle

app = FastAPI()

# Загрузка модели и scaler
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Обработчик для корневого пути
@app.get("/")
def read_root():
    return {"message": "Welcome to the Pollution Prediction API!"}

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
        # Загрузка данных
        df = pd.read_csv('pollution_data.csv')
        filtered_df = df[df['Station code'] == station_code]

        # Преобразование в временной ряд
        pivot = filtered_df.pivot(index='Measurement date', columns='Station code', values=target_variable).fillna(0)
        data = pivot.values

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
        raise HTTPException(status_code=500, detail=str(e))
# Запуск сервера
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
