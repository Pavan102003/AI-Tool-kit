import gradio as gr              
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import io
from PIL import ImageEnhance

# Image Generation Variables
API_URL_IMAGE = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
headers = {"Authorization": "Bearer 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"}

# Description Variables
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to make API request for image generation
def query_image(payload):
    response = requests.post(API_URL_IMAGE, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise ValueError(f"API request failed with status code {response.status_code}: {response.text}")

# Image Generation Function
def generate_image(prompt, brightness, contrast):
    try:
        # Query the API with the prompt
        image_bytes = query_image({"inputs": prompt})
        # Open the image from the byte stream
        image = Image.open(io.BytesIO(image_bytes))
        # Apply brightness adjustment
        brightness_enhancer = ImageEnhance.Brightness(image)
        image = brightness_enhancer.enhance(brightness)
        # Apply contrast adjustment
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(contrast)
        return image  # Return the image to display in Gradio
    except Exception as e:
        return f"Error: {e}"  # Display error message in the output

# Image Description Function
def generate_description(image):
    try:
        # Preprocess the image and generate the caption
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error: {e}"

# Main Gradio App
with gr.Blocks(css="styles.css") as demo:  # Reference the CSS file
    gr.Markdown("# AI Toolkit: Image Generation and Description")
    
    with gr.Tab("Generate Image"):
        with gr.Row():
            prompt = gr.Textbox(label="Enter Prompt")
        with gr.Row():
            brightness = gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="Brightness")
            contrast = gr.Slider(0.5, 2.0, step=0.1, value=1.0, label="Contrast")
        with gr.Row():
            generate_button = gr.Button("Generate Image")
            image_output = gr.Image()
        generate_button.click(
            generate_image,
            inputs=[prompt, brightness, contrast],
            outputs=image_output
        )

    with gr.Tab("Describe Image"):
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Image")
        with gr.Row():
            description_output = gr.Textbox(label="Generated Description")
            describe_button = gr.Button("Generate Description")
        describe_button.click(
            generate_description,
            inputs=image_input,
            outputs=description_output
        )

# Launch the Gradio interface
demo.launch()