from io import BytesIO
import torch
from PIL import Image
from flask import Flask, request, send_file
import subprocess

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
image_format = "jpg"

def run(input_image):
    image_format = "png"
    im_in = Image.open(input_image).convert("RGB")
    im_out = face2paint(model, im_in, side_by_side=False)
    output_buffer = BytesIO()
    im_out.save(output_buffer, format=image_format)
    output_buffer.seek(0)
    return output_buffer

@app.route('/face_transform', methods=['POST'])
def face_transform():
    if 'image' not in request.files:
        return 'No image provided', 400

    input_image = request.files['image']
    output_image = run(input_image)
    return send_file(output_image, mimetype=f'image/{image_format}')

@app.route('/scene_transform', methods=['POST'])
def scene_transform():
    if 'image' not in request.files:
        return 'No image provided', 400

    input_image = request.files['image']
    return image_processing_function(input_image)

def image_processing_function(input_image):
    input_image.save("dataset/test/real/image.jpg")
    cmd = "python src/infer.py --infer_dir dataset/test/real --infer_output dataset/output --ckpt_file_name checkpoints/animeganv2_generator_Shinkai.ckpt"
    subprocess.run(cmd, shell=True, check=True)
    output_image_path = "dataset\output\image.jpg"
    return send_file(output_image_path, mimetype='image/jpg')


if __name__ == '__main__':
    app.run(host='172.23.21.86', port=5000)
