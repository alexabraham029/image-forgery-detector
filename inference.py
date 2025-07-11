from PIL import Image
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and feature extractor
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=2)
model.load_state_dict(torch.load("vit_forgery_model.pth", map_location=device))
model.to(device)
model.eval()

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

def classify_image(img):
    try:
        img = Image.fromarray(img).convert("RGB")
        inputs = feature_extractor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values).logits
            pred = torch.argmax(outputs, dim=1).item()

        return "🟢 Authentic" if pred == 0 else "🔴 Forged"
    except Exception as e:
        return f"❌ Error: {str(e)}"

