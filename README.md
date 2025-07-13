# Sistema de Análisis de Satisfacción de Clientes con Lógica Difusa

[](https://opensource.org/licenses/MIT)
[](https://www.python.org/)
[](https://www.djangoproject.com/)

Aplicación web para la evaluación predictiva de la satisfacción de clientes, implementando un sistema de inferencia difusa basado en el modelo Mamdani. Esta herramienta procesa datos de entrada, aplica reglas difusas y genera análisis detallados con visualizaciones interactivas.

-----

## Características Principales

### 🧠 Motor de Inferencia Difusa

  * Implementación completa del **modelo Mamdani**.
  * **27 reglas difusas** preconfiguradas.
  * **Funciones de membresía personalizables** (triangulares y trapezoidales).
  * Sistema de **defuzzificación por método del centroide**.

### 📊 Procesamiento de Datos

  * Carga y **validación automática de archivos CSV**.
  * **Análisis estadístico descriptivo**.
  * **Normalización y preparación de datos**.
  * Detección y **manejo de valores atípicos**.

### 📈 Visualización Interactiva

  * **Gráficos de funciones de membresía**.
  * Representación de **grados de activación**.
  * Resultados de **defuzzificación**.
  * **Dashboard de análisis completo**.

### 🎓 Contenido Educativo

  * Explicaciones detalladas sobre **lógica difusa**.
  * Guía del **modelo Mamdani**.
  * **Ejemplos prácticos** de aplicación.

-----

## Requisitos del Sistema

  * **Python 3.8** o superior
  * **pip** (sistema de gestión de paquetes)
  * **Git** (control de versiones)
  * Navegador web moderno

-----

## Instalación y Configuración

### 1\. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/proyecto-logica-difusa.git
cd proyecto-logica-difusa
```

### 2\. Configurar entorno virtual (recomendado)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

### 3\. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4\. Configurar base de datos

```bash
python manage.py migrate
```

### 5\. Crear usuario administrador (opcional)

```bash
python manage.py createsuperuser
```

### 6\. Ejecutar servidor de desarrollo

```bash
python manage.py runserver
```

Acceder a la aplicación en: `http://localhost:8000`

-----

## Estructura del Proyecto

```text
proyecto-logica-difusa/
├── core/                      # Configuración principal del proyecto
│   ├── settings.py            # Configuración Django
│   ├── urls.py                # Rutas principales
│   └── ...
├── fuzzy_system/              # Aplicación de lógica difusa
│   ├── fuzzy_logic/           # Lógica de inferencia difusa
│   │   ├── membership.py      # Funciones de membresía
│   │   ├── rules.py           # Definición de reglas
│   │   └── inference.py       # Motor de inferencia
│   ├── views/                 # Controladores
│   ├── models.py              # Modelos de datos
│   ├── templates/             # Plantillas HTML
│   └── ...
├── static/                    # Archivos estáticos
├── media/                     # Archivos subidos
├── manage.py                  # Script de administración
└── requirements.txt            # Dependencias del proyecto
```

-----

## Uso del Sistema

### Flujo de Trabajo Básico

1.  **Cargar datos**: Subir un archivo CSV con los datos de clientes.
2.  **Validación**: El sistema verifica el formato y contenido.
3.  **Procesamiento**: Aplicación de reglas difusas a cada registro.
4.  **Visualización**: Generación de gráficos y análisis.
5.  **Exportación**: Opción para guardar resultados.

### Formatos de Entrada

El sistema acepta archivos CSV con las siguientes columnas mínimas:

  * `Tiempo_Suscripcion` (meses)
  * `Frecuencia_Uso` (veces/semana)
  * `Tipo_Suscripcion` (Básico/Estándar/Premium)

**Ejemplo de estructura CSV:**

```csv
ID_Cliente,Tiempo_Suscripcion,Frecuencia_Uso,Tipo_Suscripcion
1,12,5,Estándar
2,3,1,Básico
3,24,10,Premium
```

### API de Estadísticas

El sistema incluye un endpoint REST para acceder a los resultados:

```text
GET /api/stats/?customer_id=<ID>
```

**Respuesta de ejemplo:**

```json
{
    "customer_id": 1,
    "membership_degrees": {
        "bajo": 0.2,
        "medio": 0.7,
        "alto": 0.1
    },
    "predicted_satisfaction": 68.5,
    "statistical_analysis": {
        "mean": 65.3,
        "median": 67.0,
        "std_dev": 12.4
    }
}
```

-----

## Personalización

### Añadir Nuevas Reglas

Puedes editar el archivo `fuzzy_system/fuzzy_logic/rules.py` y definir nuevas reglas en el siguiente formato:

```python
new_rule = {
    'conditions': [
        ('Tiempo_Suscripcion', 'largo'),
        ('Frecuencia_Uso', 'alta'),
        ('Tipo_Suscripcion', 'Premium')
    ],
    'conclusion': ('Satisfaccion', 'muy_alta')
}
```

### Modificar Funciones de Membresía

Ajusta los parámetros en `fuzzy_system/fuzzy_logic/membership.py`:

```python
# Ejemplo para tiempo de suscripción
TIME_MEMBERSHIP = {
    'corto': {'type': 'trapmf', 'params': [0, 0, 3, 6]},
    'medio': {'type': 'trimf', 'params': [3, 12, 24]},
    'largo': {'type': 'trapmf', 'params': [12, 24, 60, 60]}
}
```

-----

## Solución de Problemas

| Problema                      | Solución                                      |
| :---------------------------- | :-------------------------------------------- |
| No se encuentra el módulo Django | `pip install django`                          |
| Puerto en uso                 | `python manage.py runserver 8080`             |
| Error en migraciones          | `python manage.py migrate --run-syncdb`       |
| Problemas con archivos CSV    | Verificar formato y columnas requeridas       |

-----

## Contribución

¡Las contribuciones son bienvenidas\! Sigue estos pasos para contribuir:

1.  Haz fork del proyecto.
2.  Crea una rama para tu nueva característica: `git checkout -b feature/awesome-feature`.
3.  Haz commit de tus cambios: `git commit -am 'Add awesome feature'`.
4.  Haz push a la rama: `git push origin feature/awesome-feature`.
5.  Abre un Pull Request.

-----

## Licencia

Distribuido bajo la **licencia MIT**. Consulta el archivo `LICENSE` para más información.

-----

## Contacto

  * Matías Sánchez - `@tu-usuario` - `email@example.com`
  * Robert Reyes - `@tu-usuario` - `email@example.com`

**Enlace del proyecto:** [https://github.com/tu-usuario/proyecto-logica-difusa](https://github.com/tu-usuario/proyecto-logica-difusa)
