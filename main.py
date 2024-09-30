import gradio as gr
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import torch
from PIL import Image

# Load model and processor
processor = ChameleonProcessor.from_pretrained("/workspace/chameleon-hf")
model = ChameleonForConditionalGeneration.from_pretrained("/workspace/chameleon-hf", torch_dtype=torch.bfloat16,
                                                          device_map="cuda")


def process_images_and_query(image_list, query, max_new_tokens):
    # Convert image paths to PIL Image objects
    image_text = '<image>'
    if image_list:
        images = [Image.open(img_path) for img_path, _ in image_list]
        diff = len(images) - len(query.count(image_text))
        query += image_text*diff
        inputs = processor(images=images, text=query, return_tensors="pt", padding=True).to(model.device,
                                                                                            dtype=torch.bfloat16)
    else:
        inputs = processor(text=query, return_tensors="pt", padding=True).to(model.device,
                                                                                            dtype=torch.bfloat16)

    # Prepare inputs


    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=int(max_new_tokens))
    output_ids = output_ids[:, inputs.input_ids.shape[-1]:]

    # Decode and return the result
    result = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return result


# Define the Gradio interface
chameleon_interface = gr.Interface(
    fn=process_images_and_query,
    inputs=[
        gr.Gallery(label="Upload Images", columns=2, rows=2, height=400, allow_preview=True),
        gr.Textbox(label="Enter your query", value="What do you see in the uploaded image?<image>"),
        gr.Number(label="Max New Tokens", value=128, minimum=1)  # New input field
    ],
    outputs=gr.Textbox(label="Model Output"),
    title="Chameleon Multi-Modal Model",
    description="Upload multiple images and enter a query to get a response from the Chameleon model.",
)

# Launch the interface
chameleon_interface.launch(server_name="0.0.0.0", server_port=5000, share=False)