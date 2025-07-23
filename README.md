# Stable Diffusion Inpainting

A web-based tool for image inpainting using Stable Diffusion and interactive mask selection powered by Segment Anything.

## Features

- Upload an image and interactively select regions to inpaint.
- Generate inpainted images using Stable Diffusion.
- Easy-to-use Gradio interface.

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:

   ```sh
   pip install -r requirements.txt
   ```

3. Download the Segment Anything model weights [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and place them in the `weights/` directory.  
   Example: `weights/sam_vit_h_4b8939.pth`
   update the `checkpoint` veriable to the relative path of the model weights

## Usage

Run the application:

```sh
python app.py
```

Open the provided local URL in your browser to access the Gradio interface.

## Notes

- The app runs on CPU by default. For GPU support, adjust the `device` variable and uncomment the `torch_dtype` line in `app.py`.
- Make sure you have the correct model weights in the `weights/` directory.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.