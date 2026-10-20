"""
app.py — Interfaz Grafica del Clasificador de Emociones
========================================================
El usuario arrastra o selecciona cualquier imagen (JPG, PNG,
BMP, WEBP, TIFF) desde CUALQUIER carpeta de su computadora.
El programa detecta el rostro y predice la emocion.

Uso:
    python app.py

Requisitos previos:
    1. Haber corrido train.py al menos una vez
    2. Tener el archivo modelo/emotion_model.h5
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
# CONFIGURACION GLOBAL
# ─────────────────────────────────────────────────────────────
# Detecta automaticamente cual modelo existe (.keras es el nuevo, .h5 el viejo)
_KERAS = os.path.join("modelo", "emotion_model.keras")
_H5    = os.path.join("modelo", "emotion_model.h5")
MODELO_PATH = _KERAS if os.path.exists(_KERAS) else _H5

CLASES  = ["Angry",   "Happy",  "Sad",    "Neutral"]
EMOJIS  = ["😠",      "😄",     "😢",     "😐"]

COLOR_EMOCION = {
    "Angry":   "#FF4D4D",
    "Happy":   "#4DCC6E",
    "Sad":     "#4D9FFF",
    "Neutral": "#AAAACC",
}

COLOR_BARRA = {
    "Angry":   "#FF4D4D",
    "Happy":   "#4DCC6E",
    "Sad":     "#4D9FFF",
    "Neutral": "#8888AA",
}

FORMATOS_IMAGEN = [
    ("Imagenes", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif"),
    ("JPG",      "*.jpg *.jpeg"),
    ("PNG",      "*.png"),
    ("BMP",      "*.bmp"),
    ("WEBP",     "*.webp"),
    ("TIFF",     "*.tiff *.tif"),
    ("Todos",    "*.*"),
]

# Paleta de colores de la UI
BG_DARK    = "#0D0D1A"
BG_PANEL   = "#13132A"
BG_CARD    = "#1A1A35"
BG_INPUT   = "#0A0A18"
TEXT_MAIN  = "#E0E0FF"
TEXT_DIM   = "#55557A"
ACCENT     = "#3A5ACC"
BORDER     = "#222244"

FONT_TITLE  = ("Segoe UI", 22, "bold")
FONT_HEAD   = ("Segoe UI", 13, "bold")
FONT_BODY   = ("Segoe UI", 12)
FONT_SMALL  = ("Segoe UI", 10)
FONT_EMOJI  = ("Segoe UI Emoji", 48)
FONT_RESULT = ("Segoe UI", 28, "bold")


# ─────────────────────────────────────────────────────────────
# APLICACION PRINCIPAL
# ─────────────────────────────────────────────────────────────
class ClasificadorApp(tk.Tk):

    def __init__(self):
        super().__init__()

        self.title("Clasificador de Emociones  🧠")
        self.geometry("1150x740")
        self.minsize(950, 650)
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        # Estado interno
        self.modelo        = None
        self.face_cascade  = None
        self.img_path      = None
        self._foto_ref     = None     # evitar garbage collection de la imagen

        self._construir_ui()
        self._cargar_modelo_async()

    # ══════════════════════════════════════
    # CONSTRUCCION DE LA UI
    # ══════════════════════════════════════
    def _construir_ui(self):
        self._barra_superior()
        self._cuerpo_principal()
        self._barra_inferior()

    # ── Barra superior ──────────────────
    def _barra_superior(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=64)
        bar.pack(fill="x", side="top")
        bar.pack_propagate(False)

        tk.Label(
            bar,
            text="⚡  Clasificador de Emociones",
            font=FONT_TITLE,
            bg=BG_PANEL, fg=TEXT_MAIN
        ).pack(side="left", padx=24, pady=14)

        # Indicador de estado del modelo
        self.var_estado = tk.StringVar(value="⏳  Cargando modelo...")
        self.lbl_estado = tk.Label(
            bar,
            textvariable=self.var_estado,
            font=FONT_SMALL,
            bg=BG_PANEL, fg="#888888"
        )
        self.lbl_estado.pack(side="right", padx=24)

        # Linea separadora
        tk.Frame(self, bg=BORDER, height=2).pack(fill="x")

    # ── Cuerpo principal (2 columnas) ───
    def _cuerpo_principal(self):
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=24, pady=16)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._panel_izquierdo(body)
        self._panel_derecho(body)

    # ── Panel izquierdo: zona de imagen ─
    def _panel_izquierdo(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL, bd=0)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        frame.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)

        # Titulo del panel
        tk.Label(
            frame,
            text="📷  Imagen de entrada",
            font=FONT_HEAD,
            bg=BG_PANEL, fg=TEXT_DIM,
            anchor="w"
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(14, 6))

        # ── Zona de previsualización ──
        canvas_frame = tk.Frame(frame, bg=BG_INPUT, bd=2, relief="flat")
        canvas_frame.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 10))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg=BG_INPUT,
            highlightthickness=0,
            cursor="hand2"
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>",  lambda e: self._seleccionar_imagen())
        self.canvas.bind("<Configure>", lambda e: self._redibujar_canvas())

        # ── Botones ──
        btn_bar = tk.Frame(frame, bg=BG_PANEL)
        btn_bar.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 14))
        btn_bar.columnconfigure(0, weight=1)

        self.btn_cargar = tk.Button(
            btn_bar,
            text="📂   Seleccionar imagen  (JPG · PNG · BMP · WEBP · TIFF)",
            font=FONT_BODY,
            bg=ACCENT, fg="white",
            activebackground="#4A6AEE",
            activeforeground="white",
            bd=0, padx=12, pady=12,
            cursor="hand2",
            relief="flat",
            command=self._seleccionar_imagen
        )
        self.btn_cargar.grid(row=0, column=0, sticky="ew")

        # Tooltip de formato
        tk.Label(
            btn_bar,
            text="Acepta imágenes desde cualquier carpeta de tu computadora",
            font=FONT_SMALL,
            bg=BG_PANEL, fg=TEXT_DIM
        ).grid(row=1, column=0, pady=(4, 0))

        # Placeholder inicial
        self.after(100, self._mostrar_placeholder)

    # ── Panel derecho: resultados ───────
    def _panel_derecho(self, parent):
        frame = tk.Frame(parent, bg=BG_PANEL)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)

        tk.Label(
            frame,
            text="🎯  Resultado del análisis",
            font=FONT_HEAD,
            bg=BG_PANEL, fg=TEXT_DIM,
            anchor="w"
        ).pack(anchor="w", padx=16, pady=(14, 8))

        # ── Tarjeta de emocion principal ──
        self.card = tk.Frame(frame, bg=BG_CARD, bd=0)
        self.card.pack(fill="x", padx=14, pady=(0, 14))

        self.lbl_emoji = tk.Label(
            self.card, text="🤔",
            font=FONT_EMOJI,
            bg=BG_CARD, fg=TEXT_MAIN
        )
        self.lbl_emoji.pack(pady=(18, 4))

        self.lbl_emocion = tk.Label(
            self.card, text="Sin analizar",
            font=FONT_RESULT,
            bg=BG_CARD, fg=TEXT_DIM
        )
        self.lbl_emocion.pack()

        self.lbl_confianza = tk.Label(
            self.card, text="",
            font=FONT_BODY,
            bg=BG_CARD, fg=TEXT_DIM
        )
        self.lbl_confianza.pack(pady=(4, 18))

        # ── Barras de probabilidad ──
        tk.Label(
            frame,
            text="📊  Probabilidades por emoción",
            font=FONT_HEAD,
            bg=BG_PANEL, fg=TEXT_DIM,
            anchor="w"
        ).pack(anchor="w", padx=16, pady=(0, 6))

        barras_frame = tk.Frame(frame, bg=BG_PANEL)
        barras_frame.pack(fill="x", padx=14)
        barras_frame.columnconfigure(1, weight=1)

        self.barras      = {}
        self.vars_barra  = {}
        self.lbls_pct    = {}

        for i, (clase, emoji) in enumerate(zip(CLASES, EMOJIS)):
            # Etiqueta
            tk.Label(
                barras_frame,
                text=f"{emoji} {clase}",
                font=FONT_BODY,
                bg=BG_PANEL, fg=TEXT_MAIN,
                width=10, anchor="w"
            ).grid(row=i, column=0, pady=6, sticky="w")

            # Barra de progreso
            style_name = f"{clase}.Horizontal.TProgressbar"
            style = ttk.Style()
            style.theme_use("clam")
            style.configure(
                style_name,
                troughcolor=BG_INPUT,
                background=COLOR_BARRA[clase],
                thickness=18,
                borderwidth=0
            )

            var = tk.DoubleVar(value=0)
            bar = ttk.Progressbar(
                barras_frame,
                variable=var,
                maximum=100,
                style=style_name,
                length=160
            )
            bar.grid(row=i, column=1, padx=(8, 8), sticky="ew", pady=6)

            lbl_pct = tk.Label(
                barras_frame,
                text="0.0%",
                font=FONT_SMALL,
                bg=BG_PANEL, fg=TEXT_DIM,
                width=6, anchor="e"
            )
            lbl_pct.grid(row=i, column=2, pady=6)

            self.barras[clase]     = bar
            self.vars_barra[clase] = var
            self.lbls_pct[clase]   = lbl_pct

        # ── Info de deteccion ──
        self.lbl_rostros = tk.Label(
            frame, text="",
            font=FONT_SMALL,
            bg=BG_PANEL, fg=TEXT_DIM
        )
        self.lbl_rostros.pack(pady=(12, 6))

        # ── Boton analizar ──
        self.btn_analizar = tk.Button(
            frame,
            text="🔍   Analizar emoción",
            font=("Segoe UI", 14, "bold"),
            bg="#2A4AAA", fg="white",
            activebackground="#3A5ABB",
            activeforeground="white",
            bd=0, padx=10, pady=14,
            cursor="hand2",
            relief="flat",
            state="disabled",
            command=self._analizar_async
        )
        self.btn_analizar.pack(fill="x", padx=14, pady=(4, 14))

    # ── Barra inferior ──────────────────
    def _barra_inferior(self):
        tk.Frame(self, bg=BORDER, height=2).pack(fill="x")
        bar = tk.Frame(self, bg=BG_PANEL, height=30)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        tk.Label(
            bar,
            text="Clasificador de Emociones  •  FER-2013 Dataset  •  CNN con TensorFlow",
            font=FONT_SMALL,
            bg=BG_PANEL, fg=TEXT_DIM
        ).pack(side="left", padx=16, pady=6)

        self.lbl_archivo = tk.Label(
            bar, text="",
            font=FONT_SMALL,
            bg=BG_PANEL, fg=TEXT_DIM
        )
        self.lbl_archivo.pack(side="right", padx=16)

    # ══════════════════════════════════════
    # PLACEHOLDER DEL CANVAS
    # ══════════════════════════════════════
    def _mostrar_placeholder(self):
        self.canvas.delete("all")
        self.canvas.update_idletasks()
        w = max(self.canvas.winfo_width(),  300)
        h = max(self.canvas.winfo_height(), 280)
        cx, cy = w // 2, h // 2

        # Fondo con puntos sutiles
        for xi in range(0, w, 30):
            for yi in range(0, h, 30):
                self.canvas.create_oval(
                    xi-1, yi-1, xi+1, yi+1,
                    fill="#16163A", outline=""
                )

        # Icono y texto
        self.canvas.create_text(
            cx, cy - 28,
            text="🖼️",
            font=("Segoe UI Emoji", 46),
            fill="#222248"
        )
        self.canvas.create_text(
            cx, cy + 38,
            text="Haz clic aquí o usa el botón de abajo\npara cargar una imagen",
            font=("Segoe UI", 12),
            fill="#333366",
            justify="center"
        )
        self.canvas.create_text(
            cx, cy + 90,
            text="JPG  •  PNG  •  BMP  •  WEBP  •  TIFF",
            font=("Segoe UI", 10),
            fill="#222248"
        )

    def _redibujar_canvas(self):
        if self.img_path and self._foto_ref:
            self._previsualizar_imagen(self.img_path)
        else:
            self._mostrar_placeholder()

    # ══════════════════════════════════════
    # CARGA DEL MODELO (hilo secundario)
    # ══════════════════════════════════════
    def _cargar_modelo_async(self):
        t = threading.Thread(target=self._cargar_modelo, daemon=True)
        t.start()

    def _cargar_modelo(self):
        try:
            # Re-verificar en tiempo de ejecucion cual modelo existe
            keras_path = os.path.join("modelo", "emotion_model.keras")
            h5_path    = os.path.join("modelo", "emotion_model.h5")

            if os.path.exists(keras_path):
                ruta = keras_path
            elif os.path.exists(h5_path):
                ruta = h5_path
            else:
                self.after(0, lambda: self._set_estado(
                    "❌  No se encontro el modelo — corre train.py primero",
                    "#FF4D4D"
                ))
                return

            self.modelo = tf.keras.models.load_model(ruta)
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            nombre = os.path.basename(ruta)
            self.after(0, lambda: self._set_estado(
                f"✅  Modelo listo  ({nombre})", "#4DCC6E"
            ))
            self.after(0, lambda: self.btn_cargar.configure(state="normal"))

        except Exception as e:
            self.after(0, lambda: self._set_estado(
                f"❌  Error: {e}", "#FF4D4D"
            ))

    def _set_estado(self, texto, color):
        self.var_estado.set(texto)
        self.lbl_estado.configure(fg=color)

    # ══════════════════════════════════════
    # SELECCION DE IMAGEN
    # ══════════════════════════════════════
    def _seleccionar_imagen(self):
        if self.modelo is None:
            messagebox.showinfo(
                "Espera",
                "El modelo aún se está cargando.\nIntenta en unos segundos."
            )
            return

        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=FORMATOS_IMAGEN
        )
        if not ruta:
            return

        self.img_path = ruta
        self.lbl_archivo.configure(text=os.path.basename(ruta))
        self._previsualizar_imagen(ruta)
        self._resetear_resultados()
        self.btn_analizar.configure(state="normal")

    def _previsualizar_imagen(self, ruta):
        try:
            img = Image.open(ruta).convert("RGB")
            self.canvas.update_idletasks()
            cw = max(self.canvas.winfo_width(),  320)
            ch = max(self.canvas.winfo_height(), 280)
            img.thumbnail((cw - 16, ch - 16), Image.LANCZOS)
            foto = ImageTk.PhotoImage(img)
            self._foto_ref = foto
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=foto)
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo leer la imagen:\n{e}")

    # ══════════════════════════════════════
    # ANALISIS (hilo secundario)
    # ══════════════════════════════════════
    def _analizar_async(self):
        self.btn_analizar.configure(state="disabled", text="⏳  Analizando...")
        t = threading.Thread(target=self._analizar, daemon=True)
        t.start()

    def _analizar(self):
        try:
            # Leer imagen con OpenCV
            img_bgr  = cv2.imread(self.img_path)
            if img_bgr is None:
                # PIL como fallback para formatos como WEBP
                img_pil  = Image.open(self.img_path).convert("RGB")
                img_bgr  = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Ecualizar histograma → mejora deteccion en fotos oscuras o con mal contraste
            img_gray_eq = cv2.equalizeHist(img_gray)

            # Deteccion de rostros con parametros mas sensibles para fotos de celular
            rostros = self.face_cascade.detectMultiScale(
                img_gray_eq,
                scaleFactor=1.05,   # mas sensible (antes 1.1)
                minNeighbors=3,     # detecta mas rostros (antes 4)
                minSize=(20, 20)    # detecta rostros mas pequenos (antes 25)
            )

            if len(rostros) == 0:
                # Sin rostro detectado usar imagen completa RGB 96x96
                face      = cv2.resize(img_gray_eq, (48, 48))
                n_rostros = 0
            else:
                # Tomar el rostro mas grande en RGB 96x96
                x, y, w, h = max(rostros, key=lambda r: r[2] * r[3])
                face      = cv2.resize(img_gray_eq[y:y+h, x:x+w], (48, 48))
                n_rostros = len(rostros)

                # Dibujar recuadro en la imagen
                img_recuadro = img_bgr.copy()
                for (rx, ry, rw, rh) in rostros:
                    cv2.rectangle(
                        img_recuadro,
                        (rx, ry), (rx + rw, ry + rh),
                        (80, 160, 255), 3
                    )
                self.after(0, lambda i=img_recuadro: self._mostrar_recuadro(i))

            # Preprocesar y predecir RGB 96x96 para MobileNetV2
            face_input = face.astype(np.float32) / 255.0
            face_input = face_input.reshape(1, 48, 48, 1)
            probs  = self.modelo.predict(face_input, verbose=0)[0]
            idx    = int(np.argmax(probs))
            emocion  = CLASES[idx]
            confianza = float(probs[idx]) * 100

            self.after(0, lambda: self._mostrar_resultados(
                emocion, confianza, probs, n_rostros
            ))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error en análisis", str(e)))
        finally:
            self.after(0, lambda: self.btn_analizar.configure(
                state="normal", text="🔍   Analizar emoción"
            ))

    # ══════════════════════════════════════
    # ACTUALIZAR UI CON RESULTADOS
    # ══════════════════════════════════════
    def _mostrar_resultados(self, emocion, confianza, probs, n_rostros):
        idx   = CLASES.index(emocion)
        color = COLOR_EMOCION[emocion]

        # Tarjeta principal
        self.lbl_emoji.configure(text=EMOJIS[idx])
        self.lbl_emocion.configure(text=emocion, fg=color)
        self.lbl_confianza.configure(
            text=f"Confianza: {confianza:.1f}%",
            fg=color
        )

        # Barras
        for i, clase in enumerate(CLASES):
            pct = float(probs[i]) * 100
            self.vars_barra[clase].set(pct)
            color_lbl = COLOR_BARRA[clase] if clase == emocion else TEXT_DIM
            self.lbls_pct[clase].configure(
                text=f"{pct:.1f}%",
                fg=color_lbl
            )

        # Info rostros
        if n_rostros == 0:
            msg = "⚠️  Sin rostro detectado — se usó la imagen completa"
        elif n_rostros == 1:
            msg = "✅  1 rostro detectado correctamente"
        else:
            msg = f"✅  {n_rostros} rostros detectados (se analizó el mayor)"
        self.lbl_rostros.configure(text=msg)

    def _mostrar_recuadro(self, img_bgr):
        """Actualiza el canvas con el recuadro azul sobre el rostro."""
        try:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            self.canvas.update_idletasks()
            cw = max(self.canvas.winfo_width(),  320)
            ch = max(self.canvas.winfo_height(), 280)
            img_pil.thumbnail((cw - 16, ch - 16), Image.LANCZOS)
            foto = ImageTk.PhotoImage(img_pil)
            self._foto_ref = foto
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, anchor="center", image=foto)
        except Exception:
            pass

    def _resetear_resultados(self):
        self.lbl_emoji.configure(text="🤔")
        self.lbl_emocion.configure(text="Sin analizar", fg=TEXT_DIM)
        self.lbl_confianza.configure(text="")
        self.lbl_rostros.configure(text="")
        for clase in CLASES:
            self.vars_barra[clase].set(0)
            self.lbls_pct[clase].configure(text="0.0%", fg=TEXT_DIM)


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ClasificadorApp()
    app.mainloop()