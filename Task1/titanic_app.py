
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="wide")


@st.cache_data
def load_data():

    for fn in ["Titanic-Dataset.csv", "titanic.csv", "Titanic.csv", "train.csv"]:
        try:
            df = pd.read_csv(fn)
            return df, fn
        except Exception:
            continue
    return None, None

def preprocess(df):
  
    raw = df.copy()


    drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")


    if "Age" in df.columns:
        df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
        df["Age"].fillna(df["Age"].median(), inplace=True)
    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        df["Fare"].fillna(df["Fare"].median(), inplace=True)
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype(str)
        df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)


    df = pd.get_dummies(df, columns=[c for c in ["Sex", "Embarked"] if c in df.columns], drop_first=True)


    y = df["Survived"].astype(int)
    X = df.drop(columns=["Survived"])

    return raw, X, y


df, used_file = load_data()
if df is None:
    st.error("Couldn't find a Titanic CSV. Put `Titanic-Dataset.csv` (or titanic.csv) in the same folder.")
    st.stop()

st.success(f"Loaded dataset: **{used_file}**")
raw_df, X, y = preprocess(df)


st.sidebar.header("Model")
model_name = st.sidebar.selectbox("Choose model", ["Logistic Regression", "Decision Tree", "Random Forest"])

models = {
    "Logistic Regression": LogisticRegression(max_iter=500, n_jobs=None),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = models[model_name]
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

left, right = st.columns([1,1])
with left:
    st.title("🚢 Titanic Survival Prediction Dashboard")
    st.markdown(f"**Selected Model:** {model_name}")
    st.markdown(f"**Accuracy:** `{acc:.2f}`")

with right:

    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax_cm,
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    ax_cm.set_title(f"Confusion Matrix — {model_name}")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)


st.header("📊 Exploratory Visualizations")

c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Survival Count")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Survived", data=raw_df, ax=ax1)
    ax1.set_xticklabels(["Died (0)", "Survived (1)"])
    st.pyplot(fig1)

with c2:
    if "Sex" in raw_df.columns:
        st.subheader("Survival by Sex")
        fig2, ax2 = plt.subplots()
        sns.countplot(x="Sex", hue="Survived", data=raw_df, ax=ax2)
        st.pyplot(fig2)

with c3:
    if "Pclass" in raw_df.columns:
        st.subheader("Survival by Class")
        fig3, ax3 = plt.subplots()
        sns.countplot(x="Pclass", hue="Survived", data=raw_df, ax=ax3)
        st.pyplot(fig3)

c4, c5 = st.columns(2)
with c4:
    if "Age" in raw_df.columns:
        st.subheader("Age Distribution")
        fig4, ax4 = plt.subplots()
        sns.histplot(raw_df["Age"].dropna(), bins=30, kde=True, ax=ax4)
        st.pyplot(fig4)

with c5:
    if "Fare" in raw_df.columns:
        st.subheader("Fare vs Survival")
        fig5, ax5 = plt.subplots()
        sns.boxplot(x="Survived", y="Fare", data=raw_df, ax=ax5)
        ax5.set_xticklabels(["Died", "Survived"])
        st.pyplot(fig5)

st.subheader("Correlation Heatmap (numeric features)")
fig_hm, ax_hm = plt.subplots(figsize=(6,4))
sns.heatmap(raw_df.select_dtypes(include=np.number).corr(), cmap="coolwarm", annot=False, ax=ax_hm)
st.pyplot(fig_hm)


if hasattr(model, "feature_importances_"):
    st.subheader("Feature Importance")
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    fig_imp, ax_imp = plt.subplots()
    sns.barplot(x=importances.values, y=importances.index, ax=ax_imp)
    ax_imp.set_xlabel("Importance")
    st.pyplot(fig_imp)


st.header("🔮 Predict Survival for a Passenger")

pclass = st.selectbox("Passenger Class (1=1st, 2=2nd, 3=3rd)", [1,2,3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard (Parch)", 0, 10, 0)
fare = st.slider("Fare", 0, 600, 50)
embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])

row = {
    "Pclass": pclass,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
}

row["Sex_male"] = 1 if sex == "male" else 0
row["Embarked_Q"] = 1 if embarked == "Q" else 0
row["Embarked_S"] = 1 if embarked == "S" else 0
user_df = pd.DataFrame([row])


user_df = user_df.reindex(columns=X.columns, fill_value=0)

if st.button("Predict"):
    pred = model.predict(user_df)[0]

    proba = float(model.predict_proba(user_df)[0][1])
    if pred == 1:
        st.success(f"✅ Likely to SURVIVE (probability: {proba:.2%})")
    else:
        st.error(f"❌ Likely to NOT survive (probability of survival: {proba:.2%})")


with st.expander("🔧 Debug info (for troubleshooting)"):
    st.write("Training columns:", list(X.columns))
    st.write("User row passed to model:", user_df)
