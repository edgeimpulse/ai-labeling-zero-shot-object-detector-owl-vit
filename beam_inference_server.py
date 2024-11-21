
from beam import Image, endpoint, Volume
from transformers import pipeline
from PIL import Image as PILImage
from io import BytesIO
import base64

# This function runs once when the container first starts
def load_models():
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device='cuda:0', cache_dir='/checkpoints')

    return detector

@endpoint(
    name="owlv2",
    cpu=1,
    memory="8Gi",
    gpu="T4",
    image=Image(
        python_version="python3.10",
        python_packages=[
            "pillow==11.0.0",
            "scikit-learn==1.5.2",
            "scikit-image==0.24.0",
            "transformers==4.45.2",
            "torch==2.5.0",
        ],
    ),
    volumes=[
        # checkpoints is used to save fine-tuned models
        Volume(name="checkpoints", mount_path="/checkpoints"),
    ],
    on_start=load_models,
    keep_warm_seconds=300
)
def predict_owlv2(context, base64_image, labels):
    detector = context.on_start_value

    im = PILImage.open(BytesIO(base64.b64decode(base64_image)))

    predictions = detector(
        im,
        labels,
    )

    print(predictions)

    return { "predictions": predictions }
