import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class GeoRankingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class DataPipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        # Фичи, на которых будет учиться модель
        self.feature_cols = ['distance_meters', 'historical_wait_time_min', 'popularity_score']

    def fit_transform(self, df: pd.DataFrame):
        """Обучает скалер и возвращает нормализованные фичи и таргет"""
        X = df[self.feature_cols].values
        y = df['target'].values
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def transform(self, df: pd.DataFrame):
        """Применяет обученный скалер для инференса"""
        X = df[self.feature_cols].values
        return self.scaler.transform(X)

def prepare_loaders(data_path="data/raw_dataset.csv", batch_size=64, train_split=0.8):
    """Загружает CSV, делит на train/val и возвращает PyTorch DataLoader'ы"""
    df = pd.read_csv(data_path)
    
    # Сортируем по сессиям, чтобы при разделении на train/val сессии не разрывались
    sessions = df['session_id'].unique()
    np.random.shuffle(sessions)
    
    split_idx = int(len(sessions) * train_split)
    train_sessions = sessions[:split_idx]
    val_sessions = sessions[split_idx:]
    
    train_df = df[df['session_id'].isin(train_sessions)].copy()
    val_df = df[df['session_id'].isin(val_sessions)].copy()
    
    pipeline = DataPipeline()
    X_train, y_train = pipeline.fit_transform(train_df)
    X_val, y_val = pipeline.transform(val_df), val_df['target'].values
    
    train_dataset = GeoRankingDataset(X_train, y_train)
    val_dataset = GeoRankingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, pipeline