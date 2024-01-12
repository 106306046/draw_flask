# DRAW_FLASK

Run a server to provide api and return an AIGC image.

## Components

* **main.py**: run a server.
* **call_api.ipynb**: an example of calling api of this server.
* **draw_diffusers.py**: download aigc model by [diffusers](https://huggingface.co/docs/diffusers/index).
* **sd_api.py**: call an api of stable diffusion web UI (Not containing).

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

```
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=8001)
```

* change prompt and other params in **draw_diffusers.py**.

```
controlnet_checkpoint = "lllyasviel/control_v11f1p_sd15_depth"
stablediffusion_checkpoint = "runwayml/stable-diffusion-v1-5"

prompt = 'wooden textur in the style of pixelart'
negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting'
seed = torch.manual_seed(8877)
step = 20
guidance_scale = 10
```