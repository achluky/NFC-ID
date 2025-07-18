from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd

app = FastAPI()

class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, emb_size)
        self.item_embedding = nn.Embedding(num_items, emb_size)

        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 1)  # regresi: output rating
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        out = self.mlp(x)
        return out.squeeze()
    
ratings_df = pd.read_csv('indonesia-tourism-destination/tourism_rating.csv')
ratings_df['userId_enc'] = ratings_df['User_Id'].astype('category').cat.codes
ratings_df['movieId_enc'] = ratings_df['Place_Id'].astype('category').cat.codes

user2idx = dict(zip(ratings_df['User_Id'].unique(), ratings_df['userId_enc'].unique()))
idx2item = dict(zip(ratings_df['movieId_enc'].unique(), ratings_df['Place_Id'].unique()))

n_users = ratings_df['userId_enc'].nunique()
n_items = ratings_df['movieId_enc'].nunique()


model = NCF(n_users, n_items, emb_size=32)
model.load_state_dict(torch.load("model/ncf_model.pth", map_location='cpu'))
model.eval()

class UserInput(BaseModel):
    user_id: int
    top_k: int = 10

@app.post("/recommend/")
def recommend(input: UserInput):
    if input.user_id not in user2idx:
        return {"error": "Unknown user_id"}

    print(input)

    user_idx = torch.tensor([user2idx[input.user_id]] * n_items,  dtype=torch.long)
    item_idx = torch.tensor(list(range(n_items)),  dtype=torch.long)

    with torch.no_grad():
        scores = model(user_idx, item_idx)

    top_k_idx = torch.topk(scores, input.top_k).indices
    recommended_items = [int(idx2item[i.item()]) for i in top_k_idx]

    return {
        "user_id": input.user_id,
        "recommended_movie_ids": recommended_items
    }
