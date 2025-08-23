import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Load data
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")
df = df.dropna(subset=['Rating'])
df['Genre'] = df['Genre'].fillna("Unknown")
df['Director'] = df['Director'].fillna("Unknown")
df['Actor 1'] = df['Actor 1'].fillna("Unknown")
df['Actor 2'] = df['Actor 2'].fillna("Unknown")
df['Actor 3'] = df['Actor 3'].fillna("Unknown")
df['Year'] = df['Title'].str.extract(r'\((\d{4})\)').astype(float).fillna(df['Year'].median())

# Feature Engineering
df['Decade'] = (df['Year'] // 10 * 10).astype(int)

# Target encoding for Director & Actors
target_encodings = {
    "Director": df.groupby("Director")["Rating"].mean().to_dict(),
    "Actor 1": df.groupby("Actor 1")["Rating"].mean().to_dict(),
    "Actor 2": df.groupby("Actor 2")["Rating"].mean().to_dict(),
    "Actor 3": df.groupby("Actor 3")["Rating"].mean().to_dict()
}
df['Director_enc'] = df['Director'].map(target_encodings["Director"])
df['Actor1_enc'] = df['Actor 1'].map(target_encodings["Actor 1"])
df['Actor2_enc'] = df['Actor 2'].map(target_encodings["Actor 2"])
df['Actor3_enc'] = df['Actor 3'].map(target_encodings["Actor 3"])

# One-hot encode Genre
ohe = OneHotEncoder(sparse=False, drop='first')
genre_ohe = ohe.fit_transform(df[['Genre']])
genre_df = pd.DataFrame(genre_ohe, columns=ohe.get_feature_names_out(['Genre']))

# Combine features
X = pd.concat([genre_df, df[['Director_enc', 'Actor1_enc', 'Actor2_enc', 'Actor3_enc', 'Year', 'Decade']]], axis=1)
y = df['Rating']

# Train Random Forest
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

# Save model and encoders
joblib.dump(model, "rf_movie_model.pkl")
joblib.dump(ohe, "genre_ohe.pkl")
joblib.dump(target_encodings, "target_encodings.pkl")

print("Model and encoders saved successfully!")
