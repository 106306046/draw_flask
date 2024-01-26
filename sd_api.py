import json
import requests
import io
import base64
import cv2
from datetime import datetime
from PIL import Image

url = "http://127.0.0.1:7861"

def paints_generation(img_input):

    print('paints_generation')

    # Read Image in RGB order
    img = cv2.imread(img_input)

    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    payload = {
        "prompt": "<lora:LORA_BCI_1:1>",
        "negative_prompt": "",
        "width": 780,
        "height": 580,
        "batch_size": 4,
        "steps": 5,
        "cfg_scale": 6,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
                        "preprocessor":"scribble_pidinet",
                        "input_image": encoded_image,
                        "model": "control_sd15_scribble [fef5e48e]",
                        "control_mode": 2,
                        "weight": 2
                    }
                ]
           }
       }
    
    }

    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)
    r = response.json()
    print(r)

    return r['images']