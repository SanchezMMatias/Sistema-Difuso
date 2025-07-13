#  Modelo y prototipo predictivo para medir la satisfacción de un cliente, usando Lógica Difusa.

Una aplicación web desarrollada en Django que implementa un sistema de análisis de satisfacción utilizando lógica difusa con el modelo Mamdani. El sistema permite procesar datos desde archivos CSV y generar análisis detallados con visualizaciones interactivas.

## 🚀 Características

- **Análisis de Lógica Difusa**: Implementación del modelo Mamdani con 27 reglas difusas predefinidas
- **Procesamiento de Datos**: Carga y validación de archivos CSV con análisis automático
- **Visualizaciones**: Generación de gráficos de funciones de membresía y análisis estadísticos
- **Interfaz Interactiva**: Sección dedicada para experimentar con funciones de membresía
- **Contenido Educativo**: Información detallada sobre lógica difusa y el modelo Mamdani

## 📋 Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Git

## 🛠️ Instalación

### 1. Clonar el Repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd mi_proyecto_django
```

### 2. Crear un Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar la Base de Datos

```bash
python manage.py migrate
```

### 5. Crear un Superusuario (Opcional)

```bash
python manage.py createsuperuser
```

## 🚀 Ejecución

Para ejecutar la aplicación en modo desarrollo:

```bash
python manage.py runserver
```

La aplicación estará disponible en: `http://127.0.0.1:8000/`

![correr la app local](https://github.com/user-attachments/assets/8126b9c1-c61f-4272-9514-27c30922352c)

## 📁 Estructura del Proyecto

```
mi_proyecto_django/
├── manage.py                       # Script principal de Django
├── db.sqlite3                      # Base de datos SQLite
├── requirements.txt                # Dependencias del proyecto
├── mi_proyecto/                    # Configuración del proyecto
│   ├── settings.py                 # Configuración principal
│   ├── urls.py                     # URLs globales
│   └── ...
└── sistema_fuzzy/                  # Aplicación principal
    ├── models.py                   # Modelos de base de datos
    ├── views.py                    # Lógica de vistas
    ├── urls.py                     # URLs de la aplicación
    ├── templates/                  # Plantillas HTML
    │   ├── home.html
    │   ├── about_mamdani.html
    │   ├── membership_functions.html
    │   └── ...
    └── migrations/                 # Migraciones de BD
```
##########################
# 📚 Análisis de Librerías del Sistema Fuzzy

Este documento describe el uso de cada librería en el sistema de lógica difusa implementado en Django.

## 🔧 Librerías del Framework

### Django
```python
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse
from django.urls import reverse
from django.conf import settings as django_settings
```

**Uso en el proyecto:**
- **`render`**: Renderiza templates HTML con contexto de datos
- **`get_object_or_404`**: Obtiene objetos de la base de datos o retorna error 404
- **`redirect`**: Redirecciona a otras vistas después de operaciones
- **`messages`**: Sistema de mensajes flash para notificaciones al usuario
- **`JsonResponse`**: Retorna respuestas JSON para APIs (`api_stats`)
- **`django_settings`**: Acceso a configuraciones del proyecto (ruta del CSV)

## 📊 Librerías de Análisis de Datos

### Pandas
```python
import pandas as pd
```

**Uso específico:**
- **Carga de datos**: `pd.read_csv(csv_path)` - Lee el dataset Netflix_Userbase_Frecuencia.csv
- **Manipulación de columnas**: Renombrado y verificación de columnas requeridas
- **Análisis estadístico**: `df.describe()` para generar estadísticas descriptivas
- **Indexación**: `df.loc[max_satisfaction_idx]` para acceder a registros específicos
- **Conversión**: `df.to_dict('records')` para convertir DataFrames a diccionarios

### NumPy
```python
import numpy as np
```

**Uso específico:**
- **Rangos numéricos**: `np.arange(0, 101, 1)` para crear rangos de valores
- **Operaciones vectoriales**: `np.zeros_like()`, `np.maximum()`, `np.minimum()`
- **Agregaciones**: `np.sum()` para cálculos de defuzzificación
- **Comparaciones**: `np.isnan()` para validar resultados válidos
- **Mapeo de colores**: `np.linspace()` para gradientes de colores en gráficos

## 📈 Librerías de Visualización

### Matplotlib
```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend sin GUI para servidor
```

**Configuración global:**
```python
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
```

**Uso específico:**
- **Gráficos de líneas**: Visualización de funciones de membresía trapezoidales y triangulares
- **Subplots**: `plt.subplots(2, 2)` para crear layouts de múltiples gráficos
- **Personalización**: Colores, etiquetas, leyendas y títulos
- **Rellenos**: `ax.fill_between()` para áreas bajo las curvas
- **Marcadores**: `ax.scatter()` para puntos específicos en las gráficas
- **Exportación**: Conversión a base64 para embedding en HTML

**Tipos de gráficos generados:**
1. **Funciones de membresía**: 4 subgráficos mostrando las funciones difusas
2. **Análisis detallado**: Visualización específica de un registro con grados de membresía
3. **Gráficos de barras**: Representación de grados de membresía activos

## 🔄 Librerías de Utilidad

### Base64
```python
import base64
```

**Uso específico:**
- **Codificación de imágenes**: `base64.b64encode(plot_data).decode()` 
- **Embedding en HTML**: Convierte gráficos matplotlib a strings base64 para mostrar en templates

### IO (BytesIO)
```python
from io import BytesIO
```

**Uso específico:**
- **Buffer de memoria**: `BytesIO()` para manejar datos de imágenes en memoria
- **Optimización**: Evita escribir archivos temporales al disco
- **Pipeline de datos**: Facilita el flujo de datos entre matplotlib y base64

### OS
```python
import os
```

**Uso específico:**
- **Rutas de archivos**: `os.path.join()` para construcción de rutas multiplataforma
- **Validación de archivos**: `os.path.exists()` para verificar existencia del CSV

### JSON
```python
import json
```

**Uso específico:**
- **Respuestas API**: Manejo de datos JSON en la función `api_stats`
- **Serialización**: Conversión de datos Python a formato JSON

### Warnings
```python
import warnings
warnings.filterwarnings('ignore')
```

**Uso específico:**
- **Supresión de advertencias**: Oculta warnings de pandas/numpy durante el procesamiento

## 🧮 Algoritmos de Lógica Difusa

### Funciones de Membresía Implementadas

```python
def trapmf(x, a, b, c, d):
    """Función de membresía trapezoidal"""
    # Implementación manual sin scikit-fuzzy
    
def trimf(x, a, b, c):
    """Función de membresía triangular"""
    # Implementación manual sin scikit-fuzzy
```

**Características:**
- **Implementación nativa**: Sin dependencias externas de lógica difusa
- **Funciones trapezoidales**: Para variables con rangos amplios
- **Funciones triangulares**: Para variables con picos específicos
- **Evaluación punto a punto**: Cálculo eficiente de grados de membresía

## 🔍 Flujo de Procesamiento

### Pipeline de Datos
1. **Carga**: Pandas lee el CSV
2. **Validación**: Verificación de columnas requeridas
3. **Procesamiento**: Aplicación de reglas difusas a cada registro
4. **Visualización**: Matplotlib genera gráficos
5. **Presentación**: Django renderiza resultados en HTML

### Variables del Sistema
- **Entrada**: Tiempo de suscripción, Frecuencia de uso, Tipo de suscripción
- **Salida**: Nivel de satisfacción predicho
- **Reglas**: 27 reglas difusas implementadas

## 📋 Dependencias Requeridas

```txt
Django>=3.2
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
```

## 🚀 Optimizaciones Implementadas

- **Backend Agg**: Matplotlib sin GUI para entornos de servidor
- **Manejo de memoria**: BytesIO para procesamiento eficiente de imágenes
- **Caching**: Configuración de matplotlib para reutilización
- **Validación robusta**: Manejo de errores y datos faltantes

## 📊 Métricas del Sistema

- **Total de reglas**: 27 reglas difusas
- **Variables de entrada**: 3 (tiempo, frecuencia, suscripción)
- **Variable de salida**: 1 (satisfacción)
- **Funciones de membresía**: 9 funciones implementadas
- **Precisión del sistema**: 95.2% (valor mostrado en dashboard)
#####################


## 📊 Funcionalidades

### 1. Análisis de Satisfacción
- Carga de archivos CSV con validación automática
- Aplicación de 27 reglas difusas predefinidas
- Cálculo de grados de membresía
- Generación de estadísticas y resultados

### 2. Funciones de Membresía Interactivas
- Experimentación con diferentes parámetros
- Visualización en tiempo real
- Herramientas educativas para comprensión de lógica difusa

### 3. Información Educativa
- Explicación detallada del modelo Mamdani
- Historia y conceptos de lógica difusa
- Beneficios y aplicaciones

## 🔄 Flujo de Trabajo

1. **Carga de Datos**: El usuario carga un archivo CSV
2. **Validación**: El sistema valida la estructura y columnas
3. **Procesamiento**: Se aplican las reglas difusas a cada registro
4. **Análisis**: Se calculan estadísticas y se generan visualizaciones
5. **Resultados**: Se presenta el análisis completo al usuario

## 🛡️ Consideraciones de Seguridad

- Validación estricta de archivos CSV
- Sanitización de datos de entrada
- Manejo seguro de archivos temporales
- Protección contra inyección de código
![Despliegue y vista de aplicación web](https://github.com/user-attachments/assets/5dd7baf4-6df4-4d17-990a-e8dd7a5af0d0)


## 🐛 Solución de Problemas

### Error: "No module named 'django'"
```bash
pip install django
```

### Error: "Port already in use"
```bash
python manage.py runserver 8080
```

### Error de base de datos
```bash
python manage.py migrate --run-syncdb
```

## 📝 Desarrollo

### Agregar Nuevas Reglas Difusas
1. Modificar el archivo `views.py` en la sección de definición de reglas
2. Actualizar la documentación correspondiente
3. Ejecutar pruebas para validar el funcionamiento

### Personalizar Funciones de Membresía
1. Editar los parámetros en la vista correspondiente
2. Ajustar las visualizaciones en `templates/membership_functions.html`
   
## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo `LICENSE` para más detalles.

## 👥 Autores

- **Matías Sánchez** - Desarrollodor del sistema de lógica difusa
- **Robert Reyes** - Desarrollodor del sistema de lógica difusa

## 🙏 Agradecimientos

- Profesor Jorge Morris arredondo por su acompañamiento y tutoria constante.

---
