import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from insightface.app import FaceAnalysis
from torchvision import datasets, models, transforms
# from face_analysis import FaceAnalysis
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description='Face Detection and Gender Classification')
parser.add_argument('--input', type=str, help='Input image or video path, or "0" for webcam')
parser.add_argument('--gender_model', type=str, help='Input image or video path, or "0" for webcam')
parser.add_argument('--device',  default="cpu", help='cuda:0 or cpu ')


args = parser.parse_args()

file_extension = os.path.splitext(args.input)[1].lower()

# Model yükleme ve hazırlama
save_path = args.gender_model
app = FaceAnalysis(name="buffalo_sc",allowed_modules=['detection'])
app.prepare(ctx_id=0, det_size=(640, 640))
model = models.resnet18(pretrained=False)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)  
# Modelin ağırlıklarını yükleyin
state_dict = torch.load(save_path, map_location=args.device)
model.load_state_dict(state_dict)
model.eval().to(args.device)
# Dönüşüm fonksiyonları
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if file_extension in ['.jpg', '.png', '.jpeg']:
    image_path = args.input
    img = cv2.imread(image_path)
    # Yüz tespiti yapma
    faces = app.get(img)
    # print(faces,"faces")
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        face_img = img[y1:y2, x1:x2]

        # Görüntüyü dönüştürme
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = transform(face_img)

        face_img = face_img.unsqueeze(0).to(args.device)

        # Görüntüyü modele verin ve tahmini alın
        with torch.no_grad():
            outputs = model(face_img)
            _, preds = torch.max(outputs, 1)

            # Softmax ile olasılıklara dönüştürme
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            
            if predicted_class.item() == 0:
                gndr="Woman"
            else:
                gndr="Man"

            print(f"Tahmin edilen sınıf: {gndr}, Olasılıklar: {probabilities}")

        label = f"{gndr}, Prob: {probabilities[0, predicted_class.item()]:.4f}"
        # label = f"{gndr}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Image - Face Detection and Gender Classification', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
elif file_extension in ['.mp4', ''] or args.input == '0':
    video_path = args.input
    if video_path=='0':
        video_path = int(video_path)
    # Webcam'i başlatma
    cap = cv2.VideoCapture(video_path)
    while True:
        # Webcam'den bir frame okuma
        ret, frame = cap.read()

        if not ret:
            break

        # Yüz tespiti yapma
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)  
            x1, y1, x2, y2 = bbox
            face_img = frame[y1:y2, x1:x2]

            try:

                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_img = transform(face_img)

                # Görüntüyü modele vermek için boyutunu ayarlayın
                face_img = face_img.unsqueeze(0).to(args.device)

                # Görüntüyü modele verin ve tahmini alın
                with torch.no_grad():
                    outputs = model(face_img)
                    _, preds = torch.max(outputs, 1)

                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1)

                    if predicted_class.item() == 0:
                        gndr="Woman"
                    else:
                        gndr="Man"


                    # print(f"Tahmin edilen sınıf: {predicted_class.item()}, Olasılıklar: {probabilities}")

                # Sonuçları görselleştirme
                label = f"{gndr}, Prob: {probabilities[0, predicted_class.item()]:.4f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            except:
                pass

        cv2.imshow('video - Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()