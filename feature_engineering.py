import pandas as pd

def load_and_process_data(filepath="Titanic-Dataset.csv"):

    df = pd.read_csv(filepath)

    
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

   
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

 
    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False)
    df["Title"] = df["Title"].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df["Title"] = df["Title"].map({"Mr":0, "Miss":1, "Mrs":2, "Master":3, "Rare":4})
    df["Title"].fillna(0, inplace=True)

 
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0,12,18,35,60,80],
                             labels=["Child","Teen","YoungAdult","Adult","Senior"])


    df.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)

    return df
