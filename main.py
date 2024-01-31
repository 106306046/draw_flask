from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
from sd_api import paints_generation
from draw_diffusers import load_model, image_process
from datetime import datetime
import torch
import gc

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

PIPE = load_model()

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def return_img_base64(img):
    # 將圖片轉換為 Base64 字串
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format='JPEG')
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
    return img_base64

@app.route('/')
def hello():
    global PIPE
    PIPE = load_model
    return 'Hello world!'

@app.route('/upload', methods=['POST'])
def upload_file():
    print('upload')

    try:
        # 從 POST 請求中取得 base64 字串
        data = request.json
        base64_string = data.get('image')

        # 解碼 base64 字串成二進制資料
        image_data = base64.b64decode(base64_string)

        # 將 bytes 資料轉換為圖片
        img = Image.open(io.BytesIO(image_data)).convert('RGB')

        # 儲存圖片
        img.save('uploads/'+datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")+'.jpg', 'JPEG')
        
        generated_img = image_process(pipe = PIPE, image= img)
        generated_img.save('outputs/'+datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p")+'.jpg','JPEG')
        base64_generated_img = return_img_base64(generated_img)

        return jsonify({'base64_image': base64_generated_img})

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/upload_imgs', methods=['POST'])
def upload_file_2():

    print('upload_imgs')

    # del PIPE
    # PIPE = None
    torch.cuda.empty_cache()
    gc.collect()

    try:
        
        # 從 POST 請求中取得 base64 字串
        data = request.json
        base64_string = data.get('image')

        # 解碼 base64 字串成二進制資料
        image_data = base64.b64decode(base64_string)

        # 將 bytes 資料轉換為圖片
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 儲存圖片
        img_path = 'uploads/'+datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p") + '_4_return' +'.jpg'
        
        img.save(img_path, 'JPEG')
        
        generated_img = paints_generation(img_path)
        base64_generated_img = [""] * 4
        
        for i in range(4):
            try:
                image = Image.open(io.BytesIO(base64.b64decode(generated_img[i])))
                image.save('outputs/' + datetime.now().strftime("%Y_%m_%d_%I_%M_%S_%p") + str(i+1) + '.jpg', 'JPEG')
                base64_generated_img[i] = return_img_base64(image)
            except Exception as e:
                print(f"Error saving or processing image {i}: {e}")
        
        return jsonify({'base64_image1': base64_generated_img[0],
                        'base64_image2': base64_generated_img[1],
                        'base64_image3': base64_generated_img[2],
                        'base64_image4': base64_generated_img[3]})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host='192.168.0.232', debug=True, use_reloader=False, port=8001)