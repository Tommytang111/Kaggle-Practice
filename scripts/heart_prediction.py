#Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#Data
df = pd.read_csv('../data/heart.csv')

#No need for data cleansing, data imputation, or data balancing for this dataset.
#Separate into features and labels
X = df.copy()
y = X.pop('HeartDisease')

#Separate by feature type
features_num = X.select_dtypes(include=['int', 'float']).columns
features_cat = X.select_dtypes(include=['object', 'string']).columns

#Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Define preprocessing
preprocessor = ColumnTransformer([('numerical', StandardScaler(), features_num), ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat)])

#Apply preprocessing
print(f"Shape before: {X_train.shape}")
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)
print(f"Shape after: {X_train.shape}")

#Training
import torch
from tqdm import tqdm
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

print(X_train_t.shape[1])
model = nn.Sequential(
    nn.Linear(X_train_t.shape[1], 32),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16,1)
)
print(model.parameters())

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(50)):
    model.train()
    for xb, yb in train_loader:
        logits = model(xb)
        loss = loss_fn(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(X_test_t), y_test_t).item()
        print(f"epoch {epoch+1} test loss {test_loss:.4f}")
              
model.eval()
with torch.no_grad():
    probs = torch.sigmoid(model(X_test_t))
    preds = (probs >0.5).float()
    acc = (preds == y_test_t).float().mean().item()
print(f"test accuracy: {acc:.3f}")
    
