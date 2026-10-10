import base64
import io
from PIL import Image


def get_reference_image_data_url():
    ref_image = Image.open(r"./data/reference_original.jpg")
    buffered = io.BytesIO()
    ref_image.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode(
        "utf-8"
    )
