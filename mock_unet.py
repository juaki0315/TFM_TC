# mock_unet.py

def crop_image(image, model=None):
    print("[🔧 MOCK] crop_image(): imagen devuelta sin cambios.")
    return image

class Humerus_model:
    @staticmethod
    def load_from_checkpoint(*args, **kwargs):
        print("[⚠️ MOCK] Humerus_model.load_from_checkpoint() llamado.")
        return Humerus_model()

    def eval(self):
        print("[⚠️ MOCK] Modelo en modo eval (sin efecto real).")
        return self
