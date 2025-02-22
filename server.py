from flask import Flask, render_template, request, jsonify, json
from werkzeug.utils import secure_filename
import os
import torch
import cv2
import face_recognition
from torch import nn
from torchvision import models, transforms
from torch.utils.data import Dataset
import numpy as np

# Define paths for uploaded files
UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

# Flask app setup
app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Model Architecture Class Definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Normalization Parameters for Image Preprocessing
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
inv_normalize = transforms.Normalize(mean=-1 * np.divide(mean, std), std=np.divide([1, 1, 1], std))

# Image Convert Function
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    cv2.imwrite('./2.png', image * 255)
    return image

# Prediction Function
def predict(model, img, path='./'):
    fmap, logits = model(img.to())
    sm = nn.Softmax()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction: ', confidence)
    return [int(prediction.item()), confidence]

# Validation Dataset Class
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        if not vidObj.isOpened():
            raise ValueError(f"Could not open video file {path}")
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
            else:
                break

# Video Detection Function
def detectFakeVideo(videoPath):
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    path_to_videos = [videoPath]

    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
    model = Model(2)
    path_to_model = 'df_model.pt'
    
    if not os.path.exists(path_to_model):
        raise FileNotFoundError('Model file not found: df_model.pt')
    
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    
    for i in range(0, len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model, video_dataset[i], './')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction

# Flask Route for Home
@app.route('/', methods=['POST', 'GET'])
def homepage():
    if request.method == 'GET':
        return render_template('index.html')
    return render_template('index.html')

# Flask Route for Detecting Fake Videos
@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        try:
            video = request.files['video']
            print(f"Received video: {video.filename}")
            video_filename = secure_filename(video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
            video.save(video_path)
            
            # Detect the video
            prediction = detectFakeVideo(video_path)
            print(f"Prediction: {prediction}")
            
            # Determine output and confidence
            if prediction[0] == 0:
                output = "FAKE"
            else:
                output = "REAL"
            
            confidence = prediction[1]
            data = {'output': output, 'confidence': confidence}
            data = json.dumps(data)
            
            # Clean up the uploaded video
            os.remove(video_path)
            
            return render_template('index.html', data=data)
        except Exception as e:
            print(f"Error processing the video: {e}")
            return "Error processing the video", 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=3000)
