import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from fastapi import FastAPI
import uvicorn

def load_data(file_path):
    """
    Carga el dataset desde un archivo CSV y muestra información básica.
    """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """
    Preprocesa los datos codificando variables categóricas y escalando valores numéricos.
    """
    df = df.copy()
    df = df.dropna()
    categorical_cols = ['gender', 'education', 'job_title', 'description']
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)
    
    scaler = StandardScaler()
    numeric_cols = ['age', 'years_experience']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def train_advanced_model(df):
    """
    Entrena un modelo más avanzado utilizando Random Forest Regressor.
    """
    X = df.drop(columns=['salary'])
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns

# Cargar y entrenar el modelo
file_path = "data.csv"
df = load_data(file_path)
df = preprocess_data(df)
advanced_model, feature_names = train_advanced_model(df)

# Crear API con FastAPI
app = FastAPI()

@app.post("/predict")
def predict_salary(data: dict):
    """
    Recibe datos en formato JSON y devuelve la predicción de salario.
    """
    try:
        input_data = pd.DataFrame([data])
        input_data = input_data.reindex(columns=feature_names, fill_value=0)
        prediction = advanced_model.predict(input_data)[0]
        return {"predicted_salary": prediction}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
