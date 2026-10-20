"""
train.py - Clasificador de Emociones (MULTI-DATASET + CNN MEJORADA)
====================================================================
Usa CNN propia optimizada para imagenes de 48x48 en escala de grises.
Mucho mejor que Transfer Learning para el dataset FER-2013.

Mejoras sobre la version anterior:
  - 4 bloques convolucionales (antes 3)
  - GlobalAveragePooling en lugar de Flatten
  - patience=20 para mas oportunidad de mejorar
  - EPOCHS=100
  - AdamW con weight decay para menos overfitting

Uso:
    python train.py
"""

import os
import zipfile
import warnings
import shutil
import glob
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ─────────────────────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────────────────────
ZIP_PATHS           = ["fer2013.zip", "FER-2012.ZIP", "fer2012.zip", "fer2013.ZIP"]
UNIFIED_DATA_FOLDER = "unified_data"
MODELO_DIR          = "modelo"

CLASES    = ["angry", "happy", "sad", "neutral"]
IMG_SIZE  = (48, 48)   # tamano original de FER-2013, optimo para esta CNN
BATCH     = 64
EPOCHS    = 100        # mas epocas con EarlyStopping que para cuando ya no mejora


# ─────────────────────────────────────────────────────────────
# 1. EXTRAER FER2013 ZIP
# ─────────────────────────────────────────────────────────────
def extraer_datos_fer2013():
    data_folder = "fer2013_data"
    if os.path.exists(data_folder):
        print("[OK] fer2013_data ya existe")
        return data_folder

    zip_file = None
    for zp in ZIP_PATHS:
        if os.path.exists(zp):
            zip_file = zp
            break

    if zip_file is None:
        import re
        for cand in glob.glob("*.zip"):
            if re.search(r"(?i)fer[-_\d]*201[23]", cand.replace("-","").replace("_","")):
                zip_file = cand
                print("[?] Detectado: " + cand)
                break

    if zip_file:
        print("[FILE] Extrayendo " + zip_file + "...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(data_folder)
        print("[OK] Extraccion completada")
    else:
        print("[!] fer2013.zip no encontrado, usando otros datasets...")

    return data_folder


def encontrar_carpetas_en_dataset(dataset_path, buscar_train_test=True):
    if not os.path.exists(dataset_path):
        return None
    for root, dirs, _ in os.walk(dataset_path):
        if "train" in dirs:
            if buscar_train_test and "test" in dirs:
                return os.path.join(root, "train"), os.path.join(root, "test")
            elif not buscar_train_test and "validation" in dirs:
                return os.path.join(root, "train"), os.path.join(root, "validation")
    return None


def copiar_imagenes(src_dir, dst_dir, emocion_dst, emocion_src=None):
    if emocion_src is None:
        emocion_src = emocion_dst
    src = os.path.join(src_dir, emocion_src)
    if not os.path.exists(src):
        return 0
    dst = os.path.join(dst_dir, emocion_dst)
    os.makedirs(dst, exist_ok=True)
    count = 0
    for f in os.listdir(src):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            dst_f = os.path.join(dst, f)
            if not os.path.exists(dst_f):
                shutil.copy2(os.path.join(src, f), dst_f)
                count += 1
    return count


# ─────────────────────────────────────────────────────────────
# 2. LIMPIAR IMAGENES CORRUPTAS
# ─────────────────────────────────────────────────────────────
def limpiar_imagenes_corruptas(carpeta):
    print("\n[*] Verificando imagenes corruptas...")
    eliminadas = 0
    revisadas  = 0
    for patron in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.pgm"]:
        for img_path in glob.glob(os.path.join(carpeta, "**", patron), recursive=True):
            revisadas += 1
            try:
                with PILImage.open(img_path) as img:
                    img.verify()
            except Exception:
                try:
                    os.remove(img_path)
                    eliminadas += 1
                except Exception:
                    pass
    print("   Revisadas  : " + str(revisadas))
    print("   Eliminadas : " + str(eliminadas))
    if eliminadas == 0:
        print("   [OK] Sin imagenes corruptas")


# ─────────────────────────────────────────────────────────────
# 3. CONSOLIDAR TODOS LOS DATASETS
# ─────────────────────────────────────────────────────────────
def consolidar_datasets():
    if os.path.exists(UNIFIED_DATA_FOLDER):
        print("[X] Limpiando " + UNIFIED_DATA_FOLDER + "...")
        shutil.rmtree(UNIFIED_DATA_FOLDER)

    train_dir = os.path.join(UNIFIED_DATA_FOLDER, "train")
    test_dir  = os.path.join(UNIFIED_DATA_FOLDER, "test")

    for clase in CLASES:
        os.makedirs(os.path.join(train_dir, clase), exist_ok=True)
        os.makedirs(os.path.join(test_dir,  clase), exist_ok=True)

    total = {c: 0 for c in CLASES}

    # FER2013
    print("\n[>] FER2013...")
    resultado = encontrar_carpetas_en_dataset("fer2013_data", True)
    if resultado:
        fer_train, fer_test = resultado
        for emocion in CLASES:
            n = copiar_imagenes(fer_train, train_dir, emocion)
            if n:
                total[emocion] += n
                print("   " + emocion.ljust(10) + " train: " + str(n))
            n = copiar_imagenes(fer_test, test_dir, emocion)
            if n:
                print("   " + emocion.ljust(10) + " test : " + str(n))
    else:
        print("   [!] No encontrado")

    # Face Expression Recognition
    print("\n[>] Face Expression Recognition Dataset...")
    resultado = encontrar_carpetas_en_dataset("Face expression recognition dataset", False)
    if resultado:
        expr_train, expr_val = resultado
        mapeo = {"angry": "angry", "happy": "happy", "sad": "sad", "neutral": "neutral"}
        for src, dst_folder in [(expr_train, train_dir), (expr_val, test_dir)]:
            for src_e, dst_e in mapeo.items():
                n = copiar_imagenes(src, dst_folder, dst_e, src_e)
                if n:
                    total[dst_e] += n
                    print("   " + src_e.ljust(10) + ": " + str(n))
    else:
        print("   [!] No encontrado")

    # Facial Emotion Recognition Image Dataset
    print("\n[>] Facial Emotion Recognition Image Dataset...")
    facial_path = "Facial Emotion Recognition Image Dataset/dataset"
    if os.path.exists(facial_path):
        mapeo = {"Angry": "angry", "Happy": "happy", "Sad": "sad", "Neutral": "neutral"}
        for src_e, dst_e in mapeo.items():
            src = os.path.join(facial_path, src_e)
            if not os.path.isdir(src):
                continue
            imgs = [f for f in os.listdir(src)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            split = int(len(imgs) * 0.8)
            for i, img in enumerate(imgs):
                dst_folder = train_dir if i < split else test_dir
                dst_f = os.path.join(dst_folder, dst_e, img)
                if not os.path.exists(dst_f):
                    shutil.copy2(os.path.join(src, img), dst_f)
            total[dst_e] += len(imgs)
            print("   " + src_e.ljust(10) + ": " + str(split) + " train + " + str(len(imgs)-split) + " test")
    else:
        print("   [!] No encontrado")

    # CK+48
    print("\n[>] CK+48...")
    if os.path.exists("CK+48"):
        mapeo = {"anger": "angry", "happy": "happy", "sadness": "sad"}
        for src_e, dst_e in mapeo.items():
            src = os.path.join("CK+48", src_e)
            if not os.path.isdir(src):
                continue
            imgs = [f for f in os.listdir(src)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.pgm'))]
            split = int(len(imgs) * 0.8)
            for i, img in enumerate(imgs):
                dst_folder = train_dir if i < split else test_dir
                dst_f = os.path.join(dst_folder, dst_e, "ck_" + src_e + "_" + img)
                if not os.path.exists(dst_f):
                    shutil.copy2(os.path.join("CK+48", src_e, img), dst_f)
            total[dst_e] += len(imgs)
            print("   " + src_e.ljust(10) + ": " + str(split) + " train + " + str(len(imgs)-split) + " test")
    else:
        print("   [!] No encontrado")

    print("\n[==] RESUMEN:")
    grand_total = 0
    for clase in CLASES:
        print("   " + clase.ljust(10) + ": " + str(total[clase]))
        grand_total += total[clase]
    print("   TOTAL     : " + str(grand_total))

    limpiar_imagenes_corruptas(UNIFIED_DATA_FOLDER)
    return train_dir, test_dir


# ─────────────────────────────────────────────────────────────
# 4. GENERADORES DE DATOS
# ─────────────────────────────────────────────────────────────
def crear_generadores(train_dir, test_dir):
    gen_train = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        brightness_range=[0.7, 1.3],
        shear_range=0.1,
        validation_split=0.15
    )
    gen_test = ImageDataGenerator(rescale=1./255)

    train_gen = gen_train.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",    # gris 48x48 — optimo para este dataset
        classes=CLASES,
        class_mode="categorical",
        batch_size=BATCH,
        subset="training",
        shuffle=True,
        seed=42
    )
    val_gen = gen_train.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        classes=CLASES,
        class_mode="categorical",
        batch_size=BATCH,
        subset="validation",
        shuffle=False,
        seed=42
    )
    test_gen = gen_test.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        classes=CLASES,
        class_mode="categorical",
        batch_size=BATCH,
        shuffle=False
    )
    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────
# 5. ARQUITECTURA CNN MEJORADA
# ─────────────────────────────────────────────────────────────
def construir_modelo():
    modelo = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3,3), padding="same", activation="relu",
                      input_shape=(48, 48, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Bloque 2
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Bloque 3
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), padding="same", activation="relu"),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        # Bloque 4 — detecta patrones mas complejos
        layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), padding="same", activation="relu"),
        layers.GlobalAveragePooling2D(),  # menos overfitting que Flatten
        layers.Dropout(0.30),

        # Clasificador
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.50),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(4, activation="softmax")
    ])

    modelo.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.001,
            weight_decay=0.0001    # regularizacion extra contra overfitting
        ),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    total = modelo.count_params()
    print("   Parametros totales: " + str(total))
    return modelo


# ─────────────────────────────────────────────────────────────
# 6. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────
def entrenar(modelo, train_gen, val_gen):
    os.makedirs(MODELO_DIR, exist_ok=True)

    class_weights = {
        0: 1.80,   # angry
        1: 1.00,   # happy
        2: 1.30,   # sad
        3: 1.30    # neutral
    }

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=20,               # mas paciencia para seguir mejorando
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODELO_DIR, "emotion_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    print("\n[>>] Entrenando CNN mejorada...")
    print("     Epocas maximas : " + str(EPOCHS))
    print("     EarlyStopping  : patience=20\n")

    history = modelo.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )
    return history


# ─────────────────────────────────────────────────────────────
# 7. EVALUACION Y GRAFICAS
# ─────────────────────────────────────────────────────────────
def evaluar(modelo, test_gen, history):
    print("\n[*] Evaluando en conjunto de prueba...")
    loss, acc = modelo.evaluate(test_gen, verbose=1)
    print("\n[OK] Test Accuracy : " + str(round(acc * 100, 2)) + "%")
    print("     Test Loss     : " + str(round(loss, 4)))

    test_gen.reset()
    y_pred  = np.argmax(modelo.predict(test_gen, verbose=0), axis=1)
    y_true  = test_gen.classes
    nombres = list(test_gen.class_indices.keys())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=nombres))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Curvas de Aprendizaje", fontsize=14, fontweight="bold")
    axes[0].plot(history.history["accuracy"],     label="Train",     linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Validacion",linewidth=2)
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["loss"],     label="Train",     linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Validacion",linewidth=2)
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELO_DIR, "curvas.png"), dpi=150)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nombres, yticklabels=nombres)
    plt.title("Matriz de Confusion")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(os.path.join(MODELO_DIR, "matriz_confusion.png"), dpi=150)
    print("[OK] Graficas guardadas en modelo/")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("   CLASIFICADOR DE EMOCIONES")
    print("   MULTI-DATASET + CNN MEJORADA")
    print("=" * 60)

    print("\n[*] PASO 1: CONSOLIDANDO DATASETS...")
    print("-" * 60)
    extraer_datos_fer2013()
    train_dir, test_dir = consolidar_datasets()

    print("\n[*] PASO 2: CREANDO GENERADORES...")
    print("-" * 60)
    train_gen, val_gen, test_gen = crear_generadores(train_dir, test_dir)
    print("\n   Clases : " + str(train_gen.class_indices))
    print("   Train  : " + str(train_gen.samples) + " imagenes")
    print("   Val    : " + str(val_gen.samples) + " imagenes")
    print("   Test   : " + str(test_gen.samples) + " imagenes")

    print("\n[*] PASO 3: CONSTRUYENDO MODELO...")
    print("-" * 60)
    modelo = construir_modelo()
    modelo.summary()

    print("\n[*] PASO 4: ENTRENANDO...")
    print("-" * 60)
    history = entrenar(modelo, train_gen, val_gen)

    print("\n[*] PASO 5: EVALUANDO...")
    print("-" * 60)
    evaluar(modelo, test_gen, history)

    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO")
    print("=" * 60)
    print("   Modelo : modelo/emotion_model.keras")
    print("   Siguiente paso : python app.py")
    print("=" * 60)