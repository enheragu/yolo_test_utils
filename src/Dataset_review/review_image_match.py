import cv2
import numpy as np

# Función para actualizar la imagen mostrada con la superposición
def update_overlay():
    global x_offset, y_offset, alpha

    # Crear una copia de la imagen RGB para superponer
    overlay = rgb_image.copy()

    # Calcular las coordenadas del recorte de la imagen en escala de grises
    x_start = max(0, x_offset - gray_width // 2)
    x_end = min(rgb_width, x_offset + gray_width // 2)
    y_start = max(0, y_offset - gray_height // 2)
    y_end = min(rgb_height, y_offset + gray_height // 2)

    # Calcular las coordenadas correspondientes en la imagen en escala de grises
    gray_x_start = max(0, gray_width // 2 - x_offset)
    gray_x_end = gray_x_start + (x_end - x_start)
    gray_y_start = max(0, gray_height // 2 - y_offset)
    gray_y_end = gray_y_start + (y_end - y_start)

    # Superponer la imagen en escala de grises con transparencia
    overlay[y_start:y_end, x_start:x_end] = cv2.addWeighted(
        overlay[y_start:y_end, x_start:x_end], 1 - alpha,
        cv2.cvtColor(
            gray_image[gray_y_start:gray_y_end, gray_x_start:gray_x_end],
            cv2.COLOR_GRAY2BGR
        ), alpha, 0
    )

    # Mostrar la imagen actualizada
    cv2.imshow('Overlay', overlay)
    print(f"Desfase actual: ({x_offset - rgb_width // 2}, {y_offset - rgb_height // 2})")

# Callback para manejar los eventos de las teclas
def on_key(event):
    global x_offset, y_offset
    if event == 27:  # Tecla 'Esc' para salir
        cv2.destroyAllWindows()
        exit()
    elif event == ord('w'):  # Arriba
        y_offset -= 1
    elif event == ord('s'):  # Abajo
        y_offset += 1
    elif event == ord('a'):  # Izquierda
        x_offset -= 1
    elif event == ord('d'):  # Derecha
        x_offset += 1
    update_overlay()

# Cargar las imágenes
lwir_image_path = '/home/arvc/eeha/kaist-cvpr15/images/set00/V000/lwir/I01689.jpg'
visible_image_path = lwir_image_path.replace('/lwir/', '/visible/')

rgb_image = cv2.imread(visible_image_path)  # Imagen en color
gray_image = cv2.imread(lwir_image_path, cv2.IMREAD_GRAYSCALE)  # Imagen en escala de grises

# Dimensiones de las imágenes
rgb_height, rgb_width = rgb_image.shape[:2]
gray_height, gray_width = gray_image.shape[:2]

# Centro inicial de la imagen en escala de grises
x_offset, y_offset = rgb_width // 2, rgb_height // 2

# Transparencia inicial
alpha = 0.5

def on_trackbar(val):
    global alpha
    alpha = val / 100
    update_overlay()

# Crear ventana
cv2.namedWindow('Overlay')

# Crear slider para ajustar la transparencia
cv2.createTrackbar('Transparencia', 'Overlay', int(alpha * 100), 100, on_trackbar)

# Mostrar la imagen inicial
update_overlay()

# Esperar eventos de teclas
while True:
    key = cv2.waitKey(0)
    on_key(key)
