import os
import sys
import cv2
import pickle as pk
import torch
from dataprocess import dealwithimage
from torchvision import transforms

from const import IMAGE_SIZE
from model import Network

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((1.1618,), (1.1180,))])

model = Network(7)
model.load_state_dict(torch.load("./saved_model/params.pth"))


haar = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

id2label = pk.load(open("./saved_model/id2label.pkl", "rb"))

def main(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    if len(faces) == 0:
        print("No faces found!")
    for x,y,w,h in faces:
        face = img[y:y+h, x:x+w]
        face = dealwithimage(face, IMAGE_SIZE, IMAGE_SIZE)
        imface = transform(face)
        imface = imface.unsqueeze(0)
        output = model(imface)
        pred = output.data.max(1, keepdim=True)[1]
        print(id2label[pred.item()])

if __name__ == "__main__":
    image_path = sys.argv[1]
    main(image_path)

