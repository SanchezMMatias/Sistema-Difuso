# Sistema de Análisis Difuso con Django

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

## 🔧 Dependencias Principales

- **Django**: Framework web principal
- **NumPy**: Cálculo numérico para operaciones difusas
- **Pandas**: Procesamiento y análisis de datos CSV
- **Matplotlib**: Generación de gráficos y visualizaciones

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
