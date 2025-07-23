import numpy as np
import torch
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image

checkpoint = "weights\sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = 'cpu'
sam = sam_model_registry[model_type](checkpoint = checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    # torch_dtype=torch.float16, # not compatable with cpu but can be used with gpu(cuda)
)

pipe.to(device)

selected_pixels = []

with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        mask_image = gr.Image(type="numpy", label="Mask Image")
        output_image = gr.Image(type="pil", label="Output Image")
    with gr.Row():
        prompt = gr.Textbox(label="Prompt")
        negative_prompt = gr.Textbox(label="Negative Prompt")
    with gr.Row():
        submit = gr.Button("Submit")

        def generate_mask(image , evt: gr.SelectData):
            selected_pixels.append(evt.index)
            predictor.set_image(np.array(image))
            input_points = np.array(selected_pixels)
            input_labels = np.ones(input_points.shape[0])
            mask,_,_ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=False,
            )
            # mask =np.logical_not(mask)  # will invert the mask
            mask = Image.fromarray(mask[0, :, :] ) 
            return mask 

        def inpaint(image , mask , prompt):

            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(mask)
            image = image.resize((512,512))
            mask = mask.resize((512,512))

            output = pipe(prompt , mask_image = mask , image = image ).images[0]
            return output
        
        input_image.select(generate_mask ,[input_image] , [mask_image]) 
        submit.click(inpaint, inputs = [input_image , mask_image , prompt ] , outputs = [output_image])

    if __name__ == "__main__":
        demo.launch()
            