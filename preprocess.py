import cv2

def preprocess_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    # Aquí pones tu lógica de recorte/detección
    # Ejemplo genérico:
    cropped = img[50:-50, 50:-50]  # Simulando recorte
    return cropped
