# Importar las bibliotecas necesarias
import cv2
import numpy as np

# Inicializar la captura de video desde un archivo de video
cap = cv2.VideoCapture('Personas_final.mp4')

# Crear el objeto de sustracción de fondo KNN con un umbral de distancia más alto
mov = cv2.createBackgroundSubtractorKNN(history=800, dist2Threshold=1000, detectShadows=False)

# Deshabilitar OpenCL
cv2.ocl.setUseOpenCL(False)

# Inicializar algunas variables
entrada_detectada = False
contador_entrada = 0

while True:
    # Leer un frame del video
    ret, frame = cap.read()

    # Si no leemos el video de manera correcta, cerramos
    if not ret:
        break

    # Obtener altura y ancho del frame
    height, width, _ = frame.shape

    # Definir región de interés (ROI) centrada en el centro de la pantalla
    roi_width = int(0.4 * width)
    roi_height = int(0.5 * height)
    roi_left = width // 2 - roi_width // 2
    roi_right = roi_left + roi_width
    roi_top = height // 2 - roi_height // 2
    roi_bottom = roi_top + roi_height
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Aplicar el detector en la ROI
    mascara = mov.apply(roi)

    # Creamos una copia para detectar los contornos
    contornos = mascara.copy()

    # Buscamos los contornos
    con, jerarquia = cv2.findContours(contornos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pasamos por los contornos
    for c in con:
        # Eliminamos ruido (contornos pequeños)
        if cv2.contourArea(c) < 13000:
            continue

        # Obtenemos los límites de los contornos en la ROI
        (x, y, w, h) = cv2.boundingRect(c)

        # Dibujamos el rectángulo en toda la pantalla
        cv2.rectangle(frame, (x + roi_left, y + roi_top), (x + w + roi_left, y + h + roi_top), (0, 255, 0), 2)

        # Verificamos si la persona entra por la puerta (40% de ancho y 50% de alto centrado)
        if roi_left < x + w // 2 < roi_right and roi_top < y + h // 2 < roi_bottom: #and not entrada_detectada:
            contador_entrada += 1
            entrada_detectada = True

    # Mostramos la cámara, mascara y contornos
    cv2.putText(frame, f'Entradas: {contador_entrada}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Detección de Entradas', frame)

    # Esperar y verificar si se presiona la tecla 'Esc' para salir del bucle
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

# Liberar la captura y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
