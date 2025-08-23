

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
df.columns = df.columns.str.strip()


if 'Rating' not in df.columns:
    raise ValueError("CSV must have a 'Rating' column")
df = df.dropna(subset=['Rating'])


for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    if col not in df.columns:
        df[col] = "Unknown"
    else:
        df[col] = df[col].fillna("Unknown")


if 'Year' not in df.columns:
    df['Year'] = 2023 
else:
    df['Year'] = df['Year'].fillna(2023)


df['Decade'] = (df['Year'] // 10 * 10).astype(int)

target_encodings = {}
for col in ['Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    target_encodings[col] = df.groupby(col)['Rating'].mean().to_dict()
    df[col + '_enc'] = df[col].map(target_encodings[col])


if 'Genre' in df.columns:
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    genre_ohe = ohe.fit_transform(df[['Genre']])
    genre_df = pd.DataFrame(genre_ohe, columns=ohe.get_feature_names_out(['Genre']))
    genre_df.index = df.index  
else:
    genre_df = pd.DataFrame()
    ohe = None


feature_cols = ['Director_enc', 'Actor 1_enc', 'Actor 2_enc', 'Actor 3_enc', 'Year', 'Decade']
X = pd.concat([genre_df, df[feature_cols]], axis=1)
y = df['Rating']


model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)


joblib.dump(model, "rf_movie_model.pkl")
if ohe is not None:
    joblib.dump(ohe, "genre_ohe.pkl")
joblib.dump(target_encodings, "target_encodings.pkl")
print("Model and encoders saved successfully!")


def predict_rating(genre, director, actor1, actor2, actor3, year):
  
    if ohe is not None and genre in ohe.categories_[0]:
        genre_encoded = ohe.transform([[genre]])
    elif ohe is not None:
        genre_encoded = np.zeros((1, len(ohe.categories_[0]) - 1))
    else:
        genre_encoded = np.array([]).reshape(1,0)
    
 
    director_enc = target_encodings['Director'].get(director, np.mean(list(target_encodings['Director'].values())))
    actor1_enc = target_encodings['Actor 1'].get(actor1, np.mean(list(target_encodings['Actor 1'].values())))
    actor2_enc = target_encodings['Actor 2'].get(actor2, np.mean(list(target_encodings['Actor 2'].values())))
    actor3_enc = target_encodings['Actor 3'].get(actor3, np.mean(list(target_encodings['Actor 3'].values())))
    

    decade = year // 10 * 10
    
 
    X_new = np.hstack([genre_encoded, [[director_enc, actor1_enc, actor2_enc, actor3_enc, year, decade]]])
    

    pred = model.predict(X_new)
    return round(float(pred[0]), 2)


def predict_and_plot(genre, director, actor1, actor2, actor3, year):
    rating = predict_rating(genre, director, actor1, actor2, actor3, year)
    
 
    fig1, ax1 = plt.subplots(figsize=(8,5))
    sns.histplot(df['Rating'], bins=20, kde=True, color='skyblue', ax=ax1)
    ax1.axvline(rating, color='red', linestyle='--', label=f'Predicted: {rating}')
    ax1.set_xlabel('IMDb Rating')
    ax1.set_ylabel('Number of Movies')
    ax1.set_title('Distribution of IMDb Ratings')
    ax1.legend()
    
   
    fig2, ax2 = plt.subplots(figsize=(10,5))
    genre_avg = df.groupby('Genre')['Rating'].mean().sort_values()
    sns.barplot(x=genre_avg.index, y=genre_avg.values, palette='viridis', ax=ax2)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    ax2.set_ylabel('Average Rating')
    ax2.set_title('Average IMDb Rating by Genre')
    
    
    fig3, ax3 = plt.subplots(figsize=(8,4))
    top_directors = df.groupby('Director')['Rating'].mean().sort_values(ascending=False).head(5)
    sns.barplot(x=top_directors.values, y=top_directors.index, palette='magma', ax=ax3)
    ax3.set_xlabel('Average Rating')
    ax3.set_title('Top 5 Directors by Average IMDb Rating')
    
    return rating, fig1, fig2, fig3


interface = gr.Interface(
    fn=predict_and_plot,
    inputs=[
        gr.Textbox(label="Genre"),
        gr.Textbox(label="Director"),
        gr.Textbox(label="Actor 1"),
        gr.Textbox(label="Actor 2"),
        gr.Textbox(label="Actor 3"),
        gr.Number(label="Year")
    ],
    outputs=[
        gr.Number(label="Predicted IMDb Rating"),
        gr.Plot(label="Rating Distribution"),
        gr.Plot(label="Genre-wise Average Rating"),
        gr.Plot(label="Top 5 Directors")
    ],
    title="IMDb Movie Rating Predictor",
    description="Enter movie details to predict its IMDb rating and view related visualizations."
)

interface.launch(share=True)
