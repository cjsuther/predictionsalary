# Proyecto de Predicción de Salarios con Machine Learning y FastAPI

## Descripción
Este proyecto implementa un modelo de Machine Learning para predecir salarios basado en características como edad, experiencia, educación, género y descripción del puesto. Se utiliza **RandomForestRegressor** para la predicción y se expone un servicio API utilizando **FastAPI**.

---
## 📌 Estructura del Código

### 1️⃣ **Carga de Datos**
```python
def load_data(file_path):
```
Carga el dataset desde un archivo CSV y lo devuelve como un DataFrame de pandas.

---
### 2️⃣ **Preprocesamiento de Datos**
```python
def preprocess_data(df):
```
- Maneja valores nulos.
- Codifica variables categóricas con **OneHotEncoder**.
- Escala variables numéricas con **StandardScaler**.
- Devuelve un DataFrame listo para entrenamiento.

---
### 3️⃣ **Entrenamiento del Modelo**
```python
def train_advanced_model(df):
```
- Divide los datos en **entrenamiento y prueba**.
- Entrena un modelo de **RandomForestRegressor** con `n_estimators=100`.
- Devuelve el modelo y los nombres de las características utilizadas.

---
### 4️⃣ **Creación de la API con FastAPI**
```python
app = FastAPI()
```
Se usa **FastAPI** para exponer un endpoint `/predict` que recibe datos en formato JSON y devuelve la predicción del salario.

```python
@app.post("/predict")
def predict_salary(data: dict):
```
- Convierte los datos de entrada en un DataFrame.
- Ajusta las características al formato esperado por el modelo.
- Genera y devuelve la predicción.

---
## 🚀 Ejecución del Proyecto
### 1️⃣ **Instalar dependencias**
```sh
pip install -r requirements.txt
```

### 2️⃣ **Ejecutar el servicio**
```sh
python main.py
```

### 3️⃣ **Probar la API con `curl`**
```sh
curl -X 'POST' 'http://localhost:8000/predict' \
-H 'Content-Type: application/json' \
-d '{
    "age": 30,
    "years_experience": 5,
    "gender_Male": 1,
    "education_Master": 1,
    "job_title_Data Scientist": 1
}'
```

📌 También puedes probar la API en **http://localhost:8000/docs** con la interfaz Swagger.

---
## 📈 Mejoras Futuras
✅ Implementar validación de datos con **Pydantic**.
✅ Agregar más modelos y comparar su rendimiento.
✅ Desplegar la API en un servicio en la nube (AWS, GCP, etc.).

---
¡Gracias por usar este proyecto! 🚀