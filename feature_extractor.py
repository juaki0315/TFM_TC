import numpy as np
import cv2
import pydicom
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from pydicom.pixel_data_handlers.util import apply_windowing

# --- Recorte cuadrado fijo centrado ---
def humerus_crop(img, crop_size=400):
    h, w, _ = img.shape
    crop_size = min(crop_size, h, w)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    return img[start_h:start_h + crop_size, start_w:start_w + crop_size]

# --- Cargar extractor de features con VGG19 ---
full_model = load_model("best_model_epoch_22_val_loss_0.5731_val_acc_0.8486.h5")
vgg_model = full_model.get_layer("vgg19")
conv_output = vgg_model.get_layer("block5_conv3").output
feature_extractor = Model(inputs=vgg_model.input, outputs=conv_output)

# --- Calcular edad desde fecha DICOM ---
def calculate_age_from_dicom(birth_date):
    try:
        birth = datetime.strptime(birth_date, "%Y%m%d")
        today = datetime.today()
        return (today - birth).days // 365
    except:
        return None

# --- Preprocesamiento estilo notebook ---
def dicom_preprocess_like_notebook(dicom, final_size=512, operation='crop', invert=False):
    img = apply_windowing(dicom.pixel_array, dicom).astype(np.float32)
    if invert:
        img = np.max(img) - img
    img -= np.min(img)
    img /= np.max(img) if np.max(img) != 0 else 1.0
    img_rgb = np.stack([img] * 3, axis=-1)

    h, w, _ = img_rgb.shape
    if operation == 'crop':
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        img_cropped = img_rgb[start_h:start_h + min_dim, start_w:start_w + min_dim]
    elif operation == 'padding':
        max_dim = max(h, w)
        pad_h = (max_dim - h) // 2
        pad_w = (max_dim - w) // 2
        img_cropped = np.pad(
            img_rgb,
            ((pad_h, max_dim - h - pad_h), (pad_w, max_dim - w - pad_w), (0, 0)),
            mode='constant', constant_values=0
        )
    else:
        raise ValueError("operation must be 'crop' or 'padding'")

    img_resized = cv2.resize(img_cropped, (final_size, final_size)).astype("float32")
    return img_resized

# --- Procesamiento completo: DICOM → RGB listo para VGG ---
def preprocess_dicom_image(dicom, target_size=(512, 512)):
    img = dicom_preprocess_like_notebook(
        dicom,
        final_size=512,
        operation='crop',
        invert=False
    )

    # Debug: guardar imagen intermedia
    cv2.imwrite("debug_1_pre_crop.png", (img * 255).astype(np.uint8))

    img_cropped = humerus_crop(img, crop_size=400)

    # Debug: después del crop
    cv2.imwrite("debug_2_post_crop.png", (img_cropped * 255).astype(np.uint8))

    # Redimensionar para VGG y mantener [0,1] normalización
    img_resized = cv2.resize(img_cropped, target_size).astype("float32")

    # 🔍 Guardar debug
    cv2.imwrite("debug_3_final_resized.png", (img_resized * 255).astype(np.uint8))


    return img_resized

# --- Extraer vector de características ---
def extract_features_from_image(img_tensor):
    fmap = feature_extractor.predict(np.expand_dims(img_tensor, axis=0))
    pooled = GlobalMaxPooling2D()(fmap).numpy().squeeze()
    return pooled
