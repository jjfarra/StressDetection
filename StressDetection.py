import tkinter as tk
import cv2
import imutils
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras import models
from PIL import Image, ImageTk

# Variables globales
root = None
canvas = None
cap = None
emotionModel = None
classes = ["severo", "moderado", "leve", "nulo"]

# Función para actualizar el frame en el Canvas
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=640)
        (locs, preds) = predict_emotion(frame, faceNet, emotionModel)

        for (box, pred) in zip(locs, preds):
            (Xi, Yi, Xf, Yf) = box
            (severo, moderado, leve, nulo) = pred
            label = "{}: {:.0f}%".format(classes[np.argmax(pred)], max(severo, moderado, leve, nulo) * 100)

            cv2.rectangle(frame, (Xi, Yi-40), (Xf, Yi), (255, 0, 0), -1)
            cv2.putText(frame, label, (Xi+5, Yi-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (Xi, Yi), (Xf, Yf), (255, 0, 0), 3)

        img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        canvas.img = img
        canvas.create_image(0, 0, anchor=tk.NW, image=img)

    root.after(10, update_frame)  # Llama a la función cada 10 milisegundos

# Función para predecir emociones
def predict_emotion(frame, faceNet, emotionModel):
    # Construye un blob de la imagen
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # Realiza las detecciones de rostros a partir de la imagen
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # Listas para guardar rostros, ubicaciones y predicciones
    faces = []
    locs = []
    preds = []

    # Recorre cada una de las detecciones
    for i in range(0, detections.shape[2]):

        # Fija un umbral para determinar que la detección es confiable
        # Tomando la probabilidad asociada en la deteccion

        if detections[0, 0, i, 2] > 0.4:
            # Toma el bounding box de la detección escalado
            # de acuerdo a las dimensiones de la imagen
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (Xi, Yi, Xf, Yf) = box.astype("int")

            # Valida las dimensiones del bounding box
            if Xi < 0: Xi = 0
            if Yi < 0: Yi = 0

            # Se extrae el rostro y se convierte BGR a GRAY
            # Finalmente se escala a 224x244
            face = frame[Yi:Yf, Xi:Xf]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (48, 48))
            face2 = img_to_array(face)
            face2 = np.expand_dims(face2, axis=0)

            # Se agrega los rostros y las localizaciones a las listas
            faces.append(face2)
            locs.append((Xi, Yi, Xf, Yf))

            pred = emotionModel.predict(face2)
            preds.append(pred[0])

    return (locs, preds)

# Función para iniciar la detección
def start_detection():
    global cap, emotionModel, faceNet
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    update_frame()

# Función para detener la detección
def stop_detection():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        canvas.delete("all")

# Crear la ventana de Tkinter
root = tk.Tk()
root.title("Detección de Estres")

# Crear el Canvas para mostrar el video
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Crear los botones
start_button = tk.Button(root, text="Iniciar Detección", command=start_detection, bg="green", fg="white")
start_button.pack(side=tk.LEFT)

stop_button = tk.Button(root, text="Detener Detección", command=stop_detection, bg="red", fg="white")
stop_button.pack(side=tk.RIGHT)

# Cargar el modelo de detección de rostros
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Cargar el detector de clasificación de emociones
emotionModel = models.load_model("model/modelFIEC_v1.h5")

# Iniciar el loop de Tkinter
root.mainloop()
