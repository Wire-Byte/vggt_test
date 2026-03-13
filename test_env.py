import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda:7" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights from local file
model = VGGT()
model.load_state_dict(torch.load("checkpoints/model.pt", map_location=device))
model = model.to(device)

# Load and preprocess example images
image_names = [
    "examples/room/images/no_overlap_1.png",
    "examples/room/images/no_overlap_2.jpg",
    "examples/room/images/no_overlap_3.jpg",
]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)

print("Inference finished.")