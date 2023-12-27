import gradio as gr
import cv2
import torch
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis
from torchvision import models
import numpy as np
import torch.nn as nn
import argparse

# Argparse kullanarak GPU ID'si almak
parser = argparse.ArgumentParser(description="Face Detection and Gender Classification")
parser.add_argument('--device',  default="cpu", help='cuda:0 or cpu')
parser.add_argument('--gender_model', type=str, help='Input image or video path, or "0" for webcam')

args = parser.parse_args()


# Model yükleme ve hazırlama
save_path = args.gender_model
app = FaceAnalysis(name="buffalo_sc",allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  # binary classification (num_of_class == 2)
# Modelin ağırlıklarını yükleyin
state_dict = torch.load(save_path, map_location=args.device)
model.load_state_dict(state_dict)
# Modeli değerlendirme moduna alın ve GPU'ya taşıyın
model.eval().to(args.device)
# Dönüşüm fonksiyonları
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
app = FaceAnalysis(allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))

def predict(image):
    # Görüntü işleme ve yüz tespiti

    results_text = ""  

    faces = app.get(image)
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]

        # Görüntüyü dönüştürme ve modele verme kısmı
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((240, 240)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        face_img = transform(face_img)
        face_img = face_img.unsqueeze(0).to(args.device)

        # Model tahmini
        with torch.no_grad():
            model.eval()
            outputs = model(face_img)
            _, preds = torch.max(outputs, 1)

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            gndr = "Woman" if predicted_class.item() == 0 else "Man"

            label = f"{gndr}"
            probability = probabilities[0, predicted_class.item()].item()
            results_text += f"{gndr} with probability {probability:.4f}\n"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image,results_text

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[gr.Image(type="numpy", label="Result"), gr.Textbox(label="Predictions")],
    title="Face Detection and Gender Classification"
)

iface.launch()