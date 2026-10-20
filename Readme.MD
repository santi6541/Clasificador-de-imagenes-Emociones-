# Clasificador de Emociones 😠 😄 😢 😐

CNN que detecta 4 emociones en imágenes faciales con interfaz gráfica.

---

## Estructura del proyecto

```
clasificador_emociones/
│
├── fer2013.zip          ← pon aquí el ZIP descargado de Kaggle
├── train.py             ← entrena el modelo (correr UNA sola vez)
├── app.py               ← interfaz gráfica (correr para usar el programa)
├── requirements.txt     ← dependencias
├── README.md
│
└── modelo/              ← se crea automáticamente al entrenar
    ├── emotion_model.h5
    ├── curvas.png
    └── matriz_confusion.png
```

---

## Instalación paso a paso

### 1. Abrir terminal en VS Code
Menú → Terminal → New Terminal

### 2. Crear entorno virtual
```
python -m venv venv
```

### 3. Activar el entorno virtual
Windows:
```
venv\Scripts\activate
```
Mac / Linux:
```
source venv/bin/activate
```
Verifica que aparece `(venv)` al inicio de la terminal.

### 4. Instalar dependencias
```
pip install -r requirements.txt
```

### 5. Poner el ZIP en la carpeta raíz
El script busca automáticamente cualquiera de estos nombres:

```
fer2013.zip
FER-2012.ZIP
fer2012.zip
```

Si tu archivo tiene otro nombre, renómbralo a uno de los anteriores o mueve una copia. El contenido debe ser el dataset FER (misma estructura de carpetas `train/` y `test/`).

Puedes descargarlo de:
https://www.kaggle.com/datasets/msambare/fer2013

Colócalo en la misma carpeta que `train.py` y `app.py`.

### 6. Entrenar el modelo (solo una vez, tarda 30‑60 min en CPU)

Este paso crea el directorio `modelo/` y guarda el archivo `emotion_model.h5` que usa la interfaz.
Si ya cuentas con ese archivo puedes saltar directamente al paso 7.

```bash
python train.py
```

Durante el entrenamiento se mostrarán curvas de aprendizaje y al final se imprimirán los resultados:

```
✅  Entrenamiento completado.
   Modelo guardado en: modelo/emotion_model.h5
```

> 📌 El dataset `fer2013.zip` debe estar presente en la carpeta del proyecto.
> Puedes descargarlo manualmente desde Kaggle o usar la [Kaggle CLI](https://github.com/Kaggle/kaggle-api):
> ```bash
> kaggle datasets download -d msambare/fer2013 -p .
> ```


### 7. Correr la interfaz gráfica
Una vez entrenado el modelo basta ejecutar:

```bash
python app.py
```

Si el programa no encuentra `modelo/emotion_model.h5` mostrará un mensaje de error y saldrá; en ese caso vuelve al paso 6.

> ⚠️  La primera vez el cargado del modelo puede tardar unos segundos. El botón *Seleccionar imagen* permanecerá deshabilitado hasta que aparezca el aviso **"✅ Modelo listo"**.


---

## Verificar el entorno (opcional)
Si quieres comprobar que las dependencias se instalaron correctamente puedes ejecutar lo siguiente en Python:

```python
import tensorflow as tf, cv2, numpy as np
print(tf.__version__, cv2.__version__, np.__version__)
```

Deberías ver las versiones de las librerías sin errores de importación.

---

## Cómo usar la interfaz

1. Espera a que aparezca **"✅ Modelo listo"** arriba a la derecha
2. Haz clic en **"Seleccionar imagen"** o en la zona de la imagen
3. Elige cualquier foto desde cualquier carpeta de tu computadora
4. Haz clic en **"Analizar emoción"**
5. Ve el resultado con la emoción detectada y las barras de probabilidad

Formatos aceptados: JPG · PNG · BMP · WEBP · TIFF

---

## Notas

- Si tienes Python 3.13 y tensorflow no instala, usa: `pip install tensorflow-cpu`
- El modelo solo necesita entrenarse una vez. Después usa solo `app.py`.
- Para mejores resultados usa fotos con el rostro visible y bien iluminado.