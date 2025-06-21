import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def train_models():
    df = pd.read_csv("matches.csv")
    X = df[['team1_attack','team2_defense','team2_attack','team1_defense']]
    y1 = df['team1_score']; y2 = df['team2_score']
    model1 = RandomForestRegressor().fit(X, y1)
    model2 = RandomForestRegressor().fit(X, y2)
    return model1, model2

st.title("⚽ AI İle Maç Skor Tahmini")
t1_attack = st.slider("Takım 1 Hücum Gücü",0,100,85)
t1_defense = st.slider("Takım 1 Savunma Gücü",0,100,80)
t2_attack = st.slider("Takım 2 Hücum Gücü",0,100,82)
t2_defense = st.slider("Takım 2 Savunma Gücü",0,100,78)

if st.button("Tahmini Skoru Getir"):
    m1, m2 = train_models()
    X_new = [[t1_attack, t2_defense, t2_attack, t1_defense]]
    score1 = round(m1.predict(X_new)[0])
    score2 = round(m2.predict(X_new)[0])
    st.success(f"Tahmini Skor → Takım 1: {score1} – Takım 2: {score2}")team1_attack,team2_defense,team2_attack,team1_defense,team1_score,team2_score
85,78,82,80,2,1
90,83,88,85,3,2
75,70,76,72,1,1
80,74,85,76,2,2
70,65,60,72,1,0
streamlit
pandas
scikit-learn
