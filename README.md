# DRAW_FLASK

Run a server to provide API and return an AIGC image.

## Components

* **main.py**: Run a server. 
* **call_api.ipynb**: An example of calling API of this server.
* **draw_diffusers.py**: Download model by [diffusers](https://huggingface.co/docs/diffusers/index).
* **sd_api.py**: Call an API of stable diffusion web UI (Not containing).

After running **main.py**, folder **uploads** will save images sent by API and folder **outputs** will save results generated by AI model.

## Environment

1. check cuda version

```
nvcc --version
```

2. pytorch

    [link](https://pytorch.org/get-started/locally/)

    It will install **pillow** and **numpy** in the mean time.

3. diffusers

```
pip install accelerate
pip install transformers
pip install diffusers
```

4. others

```
pip install flask
pip install opencv-python
```

## Run 

> Model will be downloaded in the fist execution.


Run **main.py** by VSCode

or 

```
Python main.py
```

* change server port number in **main.py**.

```python
if __name__ == '__main__':
    app.run(host='192.168.0.232', debug=True, use_reloader=False, port=8001)
```

* change prompt and other params in **draw_diffusers.py**.

```python
controlnet_checkpoint = "lllyasviel/control_v11f1p_sd15_depth"
stablediffusion_checkpoint = "runwayml/stable-diffusion-v1-5"

prompt = 'wooden textur in the style of pixelart'
negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting'
seed = torch.manual_seed(8877)
step = 20
guidance_scale = 10
```