import pandas as pd
import numpy as np
import joblib
import gradio as gr
from sklearn.preprocessing import OneHotEncoder

# -----------------------------
# Load trained model and encoders
# -----------------------------
# Make sure you have saved your trained RandomForest model and OneHotEncoder
model = joblib.load("rf_movie_model.pkl")
ohe = joblib.load("genre_ohe.pkl")
target_encodings = joblib.load("target_encodings.pkl") 
# target_encodings should be a dict like:
# {"Director": {...}, "Actor 1": {...}, "Actor 2": {...}, "Actor 3": {...}}

# -----------------------------
# Prediction function
# -----------------------------
def predict_rating(genre, director, actor1, actor2, actor3, year):
    # One-hot encode Genre
    genre_encoded = ohe.transform([[genre]])
    
    # Target encoding for directors and actors
    director_enc = target_encodings["Director"].get(director, np.mean(list(target_encodings["Director"].values())))
    actor1_enc = target_encodings["Actor 1"].get(actor1, np.mean(list(target_encodings["Actor 1"].values())))
    actor2_enc = target_encodings["Actor 2"].get(actor2, np.mean(list(target_encodings["Actor 2"].values())))
    actor3_enc = target_encodings["Actor 3"].get(actor3, np.mean(list(target_encodings["Actor 3"].values())))
    
    # Feature Engineering: Decade
    decade = year // 10 * 10
    
    # Combine features
    X_new = np.hstack([genre_encoded, [[director_enc, actor1_enc, actor2_enc, actor3_enc, year, decade]]])
    
    # Predict rating
    rating_pred = model.predict(X_new)
    return round(float(rating_pred[0]), 2)

# -----------------------------
# Create Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict_rating,
    inputs=[
        gr.Textbox(label="Genre"),
        gr.Textbox(label="Director"),
        gr.Textbox(label="Actor 1"),
        gr.Textbox(label="Actor 2"),
        gr.Textbox(label="Actor 3"),
        gr.Number(label="Year")
    ],
    outputs=gr.Number(label="Predicted IMDb Rating"),
    title="IMDb Movie Rating Predictor",
    description="Enter movie details to predict its IMDb rating."
)

# Launch the app with a shareable link
interface.launch(share=True)
