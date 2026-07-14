import numpy as np
import pandas as pd
import h3

def generate_geospatial_data(num_sessions=1000, points_per_session=10):
    """
    Генерирует синтетические данные для ранжирования точек посадки.
    Каждая сессия содержит координаты пользователя и N потенциальных точек вокруг.
    """
    data = []
    
    # Центр симуляции (например, центр города)
    base_lat, base_lng = 43.2389, 76.8897 
    
    for session_id in range(num_sessions):
        # 1. Генерируем случайную позицию пользователя в радиусе ~2-3 км
        user_lat = base_lat + np.random.uniform(-0.02, 0.02)
        user_lng = base_lng + np.random.uniform(-0.02, 0.02)
        
        # Получаем H3 индекс для пользователя (разрешение 8 — размер ячейки ~0.7кв. км)
        user_h3 = h3.geo_to_h3(user_lat, user_lng, resolution=8)
        
        # 2. Генерируем кандидатов (pickup points) вокруг пользователя
        for point_idx in range(points_per_session):
            # Точки чуть ближе или дальше
            point_lat = user_lat + np.random.uniform(-0.005, 0.005)
            point_lng = user_lng + np.random.uniform(-0.005, 0.005)
            point_h3 = h3.geo_to_h3(point_lat, point_lng, resolution=8)
            
            # Считаем расстояние в метрах по формуле гаверсинуса (встроенная в h3)
            distance_meters = h3.point_dist((user_lat, user_lng), (point_lat, point_lng), unit='m')
            
            # Симулируем фичи инфраструктуры в этой гео-зоне
            historical_wait_time = np.random.uniform(2.0, 15.0)  # среднее время ожидания в минутах
            popularity_score = np.random.uniform(0.0, 1.0)       # популярность точки у других водителей
            
            # Формируем таргет (выбрал ли пользователь эту точку)
            # Логика: пользователь предпочитает точки ближе и с меньшим временем ожидания
            score = - (distance_meters * 0.01) - (historical_wait_time * 0.5) + (popularity_score * 2)
            noise = np.random.normal(0, 1)
            selection_prob = 1 / (1 + np.exp(-(score + noise))) # Сигмоида
            
            data.append({
                "session_id": session_id,
                "user_lat": user_lat,
                "user_lng": user_lng,
                "user_h3": user_h3,
                "point_id": f"p_{session_id}_{point_idx}",
                "point_lat": point_lat,
                "point_lng": point_lng,
                "point_h3": point_h3,
                "distance_meters": distance_meters,
                "historical_wait_time_min": historical_wait_time,
                "popularity_score": popularity_score,
                "selection_prob": selection_prob
            })
            
    df = pd.DataFrame(data)
    
    # Для каждой сессии делаем ровно одну точку "выбранной" (Target = 1) основываясь на максимальной вероятности
    df['target'] = 0
    for idx, group in df.groupby('session_id'):
        max_prob_idx = group['selection_prob'].idxmax()
        df.loc[max_prob_idx, 'target'] = 1
        
    df.drop(columns=['selection_prob'], inplace=True)
    return df

if __name__ == "__main__":
    print("⏳ Генерация гео-данных...")
    dataset = generate_geospatial_data(num_sessions=2000, points_per_session=10)
    dataset.to_csv("data/raw_dataset.csv", index=False)
    print(f"✅ Успешно сохранено {len(dataset)} строк в data/raw_dataset.csv")
    print(dataset[['session_id', 'distance_meters', 'user_h3', 'target']].head(10))