import torch
import torch.nn as nn
import torch.optim as optim
import os
from src.pipeline import prepare_loaders

# 1. Архитектура нейросети для ранжирования
class PickupPointRanker(nn.Module):
    def __init__(self, input_dim):
        super(PickupPointRanker, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)  # Выход — сырой скор (логит)
        )
        
    def forward(self, x):
        return self.net(x)

# 2. Функция обучения модели
def train_model(epochs=15, lr=0.005):
    print("📦 Подготовка данных...")
    train_loader, val_loader, pipeline = prepare_loaders()
    
    # Инициализация модели, лосса и оптимизатора
    input_dim = len(pipeline.feature_cols)
    model = PickupPointRanker(input_dim)
    
    # Так как у нас бинарная классификация (выбрал / не выбрал)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"🚀 Начало обучения модели {model.__class__.__name__}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    # Сохраняем веса модели и обученный скалер
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/ranker_weights.pth")
    
    # Сохраняем pipeline (скалер) для инференса в API
    import joblib
    joblib.dump(pipeline, "models/pipeline.pkl")
    print("🎉 Модель и пайплайн успешно сохранены в папку models/")

if __name__ == "__main__":
    train_model()