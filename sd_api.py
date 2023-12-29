import json
import requests
import io
import base64
import cv2
from PIL import Image

url = "http://127.0.0.1:7860"

def paints_generation(img_input):


    # Read Image in RGB order
    img = cv2.imread(img_input)

    # Encode into PNG and send to ControlNet
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    payload = {
        "prompt": "<lora:LORA_BCI_1:1>",
        "negative_prompt": "",
        "batch_size": 1,
        "steps": 18,
         "cfg_scale": 6,
        "alwayson_scripts": {
            "controlnet": {
                "args": [
                    {
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

    image = Image.open(io.BytesIO(base64.b64decode(r['images'][0])))
    image.save('output.png')

    return r['images'][0]