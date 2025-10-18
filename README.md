# Proyecto CNN — Clasificación CIFAR-10 🧠📊

Este proyecto implementa una **red neuronal convolucional (CNN)** para clasificar imágenes del dataset **CIFAR-10**, que contiene 10 categorías diferentes de objetos.

## 📌 Fases del Proyecto

### Fase 1 — Preparación del Dataset Visual (CIFAR-10)
- Carga del dataset CIFAR-10 desde `tensorflow.keras.datasets`.
- Conversión a escala de grises y normalización de píxeles.
- One-hot encoding de las etiquetas.
- Exploración básica de los datos.

### Fase 2 — Arquitectura del "Córtex Visual"
- Definición de la arquitectura CNN.
- Dos bloques convolucionales + MaxPooling2D.
- Capa Flatten + Dense de 64 neuronas + salida softmax.
- Compilación del modelo con Adam y categorical crossentropy.

### Fase 3 — Entrenamiento y Evaluación
- Entrenamiento con EarlyStopping para evitar sobreajuste.
- Visualización de curvas de precisión y pérdida.
- Evaluación final en el conjunto de prueba.

## 🚀 Requisitos

- Python 3.10+
- TensorFlow
- scikit-image
- Matplotlib
- NumPy

Instalación de dependencias:
```bash
pip install -r requirements.txt
```

## 🧪 Ejecución del proyecto

```bash
python entrenar_modelo.py
```

Esto entrenará el modelo y mostrará las métricas de rendimiento.

## 👥 Autores

- **Álvaro Costa Oyola** — [@Alvaroco12](https://github.com/Alvaroco12)  
- **Nicolás Amador Montesinos** — [@Niicooam12](https://github.com/Niicooam12)

## 📝 Licencia

Este proyecto es de uso académico y libre de modificación y distribución con fines educativos.
