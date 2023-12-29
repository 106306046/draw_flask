from argparse import ArgumentParser
import torch
import cv2
import io
import base64
import numpy as np
from PIL import Image, ImageOps
# from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation
from transformers import pipeline
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel, UniPCMultistepScheduler

controlnet_checkpoint = "lllyasviel/control_v11f1p_sd15_depth"
stablediffusion_checkpoint = "runwayml/stable-diffusion-v1-5"

prompt = 'wooden textur in the style of pixelart'
negative_prompt = 'worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting'
seed = torch.manual_seed(8877)
step = 20
guidance_scale = 10
    
def get_canny(image):
    
    canny_image = cv2.Canny(np.array(image), 100, 200,)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = cv2.bitwise_not(canny_image)
    canny_image = Image.fromarray(canny_image)

    return canny_image

def get_depth(image):
    
    ### 1

    # processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    # model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # # prepare image for the model
    # inputs = processor(images=image, return_tensors="pt")

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     predicted_depth = outputs.predicted_depth

    # # interpolate to original size
    # prediction = torch.nn.functional.interpolate(
    #     predicted_depth.unsqueeze(1),
    #     size=image.size[::-1],
    #     mode="bicubic",
    #     align_corners=False,
    # )

    # # visualize the prediction
    # output = prediction.squeeze().cpu().numpy()
    # formatted = (output * 255 / np.max(output)).astype("uint8")
    # depth = Image.fromarray(formatted)

    # return depth

    ### 2

    # print("get depth")
    # depth_estimator = pipeline("depth-estimation")
    # depth_map = depth_estimator(image)["depth"]
    # depth_map = np.array(depth_map)
    # depth_map = depth_map[:, :, None]
    # depth_map = np.concatenate([depth_map, depth_map, depth_map], axis=2)
    # detected_map = torch.from_numpy(depth_map).float() / 255.0
    # depth_map = detected_map.permute(2, 0, 1)

    # depth_map = depth_map.unsqueeze(0).half()

    # torch.cuda.empty_cache()

    # return depth_map

    ### 3 
    image_invert = ImageOps.invert(image)
    
    return image_invert

def load_model():

    print("get model")

    # model

    controlnet = ControlNetModel.from_pretrained(
        controlnet_checkpoint, 
        torch_dtype=torch.float16
    ).to("cuda")

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        stablediffusion_checkpoint, 
        controlnet=controlnet, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")


    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe

def image_process(pipe, image): 

    print("image process")
    
    # get canny image
    canny_image = get_canny(image)

    # get depth map
    depth_image = get_depth(image)

    # get generated image
    output_image = pipe(
        prompt = prompt, 
        negative_prompt = negative_prompt,
        image = canny_image, 
        num_inference_steps = step,
        generator = seed, 
        control_image = depth_image,
        guidance_scale = guidance_scale
    ).images[0]

    return output_image

def image_to_base64(img):
    
    output_buffer = io.BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    
    return base64_str


def main(img_path):

    image = Image.open(img_path)

    pipe = load_model()

    output = image_process(pipe=pipe,image=image)

    output_bytes = image_to_base64(output)

    return output_bytes


if __name__ == '__main__':
    
    parser = ArgumentParser(prog = "Generate AI img", description="input: img bytearry, Process, Output: AI generated img")
    
    parser.add_argument("--img_path", type=str, help="input a bytearray of init image")

    args = parser.parse_args()

    img_path = args.img_path

    print(main(img_path))
    


