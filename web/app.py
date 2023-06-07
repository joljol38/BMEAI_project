#%%
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image
import pickle
import os
from urllib.parse import quote

app = Flask(__name__, static_folder='static')

# 모델 불러오기
model1 = torch.load('resnet_binary2.pkl', map_location=torch.device('cpu'))
model2 = torch.load('resnet_multi2.pkl', map_location=torch.device('cpu'))
# 이미지 전처리 함수
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# 예측 함수
def predict(image, model):
    image = preprocess_image(image)
    outputs = model(image)
    _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

# 루트 엔드포인트
@app.route('/')
def index():
    return render_template('index.html')

# 예측 엔드포인트
@app.route('/predict', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'no image file provided'})

    file = request.files['image']
    image = Image.open(file).convert('RGB')

    model_type = request.form.get('model')
    if model_type == 'binary':
        result = predict(image, model1)
    elif model_type == 'multi':
        result = predict(image, model2)
    else:
        return jsonify({'error': 'invalid model'})

    # Encode the uploaded image filename
    uploaded_image = quote(file.filename)

    # Save the uploaded image to the static folder
    save_path = os.path.join(app.root_path, 'static', file.filename)
    image.save(save_path)

    return render_template('predict.html', model=model_type, result=result, uploaded_image=uploaded_image)


if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True, use_reloader=True)

# %%
