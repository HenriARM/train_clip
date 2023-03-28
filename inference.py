import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# for inference the conversion step from fp16 to fp32 is not needed
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device) # (B, C, H, W)
text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device) # (B, T - tokens), in this case words amount is B

with torch.no_grad():
    image_features = model.encode_image(image) # (B, E), here (1, 512)
    text_features = model.encode_text(text) # (B1, E), here (3, 512) 
    
    logits_per_image, logits_per_text = model(image, text) # (B, B1) here (1, 3)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]