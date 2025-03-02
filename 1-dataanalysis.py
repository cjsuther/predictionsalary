import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

if __name__ == "__main__":
    file_path = "data.csv"  # Cambiar por la ruta real del archivo
    df = load_data(file_path)
    check_missing_values(df)
    describe_data(df)
    plot_distributions(df)
    plot_correlations(df)
