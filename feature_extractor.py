import numpy as np
import cv2
import pydicom
from datetime import datetime
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import GlobalMaxPooling2D
from pydicom.pixel_data_handlers.util import apply_windowing
try:
    from utilidades_cropping_unet import Humerus_model, crop_image
except ImportError:
    from mock_unet import Humerus_model, crop_image

# --- Cargar extractor de features con VGG19 ---
full_model = load_model("best_model_epoch_19_val_loss_0.3395_val_acc_0.8645.h5")
vgg_model = full_model.get_layer("vgg19")
conv_output = vgg_model.get_layer("block5_conv3").output
feature_extractor = Model(inputs=vgg_model.input, outputs=conv_output)

# --- Cargar modelo Unet para segmentar húmero ---
unet_model = Humerus_model.load_from_checkpoint(
    checkpoint_path="UnetPlusPlus_efficientnet-b2_25.ckpt",
    arch="UnetPlusPlus",
    encoder_name="efficientnet-b2",
    in_channels=3,
    out_classes=1,
    map_location="cpu"
)
unet_model.eval()

# --- Calcular edad desde fecha DICOM ---
def calculate_age_from_dicom(birth_date):
    try:
        birth = datetime.strptime(birth_date, "%Y%m%d")
        today = datetime.today()
        return (today - birth).days // 365
    except:
        return None

# --- Procesamiento completo: DICOM → RGB segmentado y listo para VGG ---
def preprocess_dicom_image(dicom, target_size=(300, 300)):
    # Apply windowing
    img = apply_windowing(dicom.pixel_array, dicom).astype(np.float32)

    # Normalizar y convertir a RGB
    img -= np.min(img)
    img /= np.max(img) if np.max(img) != 0 else 1.0
    img = np.stack([img] * 3, axis=-1)

    # Recortar húmero con Unet
    img_cropped = crop_image(img, unet_model)

    # Redimensionar para VGG
    img_resized = cv2.resize(img_cropped, target_size).astype("float32") / 255.0

    return img_resized

# --- Extraer vector de características ---
def extract_features_from_image(img_tensor):
    fmap = feature_extractor.predict(np.expand_dims(img_tensor, axis=0))
    pooled = GlobalMaxPooling2D()(fmap).numpy().squeeze()
    return pooled