import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def load_data(file_path):
    """
    Carga el dataset desde un archivo CSV y muestra información básica.
    """
    df = pd.read_csv(file_path)
    print("\nInformación del Dataset:")
    print(df.info())
    print("\nPrimeras filas:")
    print(df.head())
    return df

def check_missing_values(df):
    """
    Verifica valores nulos en el dataset.
    """
    missing_values = df.isnull().sum()
    print("\nValores nulos por columna:")
    print(missing_values[missing_values > 0])

def describe_data(df):
    """
    Muestra estadísticas descriptivas de las variables numéricas y categóricas.
    """
    print("\nEstadísticas descriptivas de variables numéricas:")
    print(df.describe())
    print("\nDistribución de variables categóricas:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\n{col}:")
        print(df[col].value_counts())

def plot_distributions(df):
    """
    Genera histogramas de variables numéricas.
    """
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols].hist(figsize=(12, 8), bins=30, edgecolor='black')
    plt.suptitle("Distribuciones de Variables Numéricas", fontsize=14)
    plt.show()

def plot_correlations(df):
    """
    Muestra un heatmap con correlaciones entre variables numéricas.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Matriz de Correlación", fontsize=14)
    plt.show()

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

def train_model(df):
    """
    Entrena y valida un modelo de regresión lineal para predecir el salario.
    """
    X = df.drop(columns=['salary'])
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nEvaluación del Modelo:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    
    # Validación cruzada
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("\nValidación Cruzada (MAE Promedio):", -scores.mean())
    
    return model

def optimize_model(df):
    """
    Optimiza el modelo usando GridSearchCV.
    """
    X = df.drop(columns=['salary'])
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {'fit_intercept': [True, False]}
    grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)
    
    print("\nMejores hiperparámetros:", grid_search.best_params_)
    print("Mejor MAE en validación cruzada:", -grid_search.best_score_)
    
    return grid_search.best_estimator_

def train_advanced_model(df):
    """
    Entrena un modelo más avanzado utilizando Random Forest Regressor.
    """
    X = df.drop(columns=['salary'])
    y = df['salary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\nEvaluación del Modelo Avanzado (Random Forest):")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
    
    return model

if __name__ == "__main__":
    file_path = "data.csv"  # Cambiar por la ruta real del archivo
    df = load_data(file_path)
    check_missing_values(df)
    describe_data(df)
    plot_distributions(df)
    plot_correlations(df)
    df = preprocess_data(df)
    model = train_model(df)
    best_model = optimize_model(df)
    advanced_model = train_advanced_model(df)
