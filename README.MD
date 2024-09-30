# Chameleon Multi-Modal Model

This project uses the Chameleon Multi-Modal Model to process images and text queries. The model is capable of generating responses based on the input images and text.

## Requirements

- Python
- Gradio
- Transformers
- Torch
- PIL

## Installation

Clone the repository and navigate to the project directory. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the `main.py` script:

```bash
python main.py
```

This will launch a Gradio interface where you can upload multiple images and enter a query to get a response from the Chameleon model.

The interface includes the following input fields:

- **Upload Images**: Allows you to upload multiple images.
- **Enter your query**: Allows you to enter a text query related to the uploaded images.
- **Max New Tokens**: Allows you to specify the maximum number of new tokens for the model to generate.

The model's output will be displayed in the **Model Output** field.

## Note

The model and processor are loaded from the `/workspace/chameleon-hf` directory. Make sure to replace this with the correct path to your model and processor.


## License

[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Discord Server

Join our Discord server [here](https://discord.gg/xhcBDEM3).

## Feeling Generous? 😊

Eager to buy me a cup of $2 coffee or iced tea? 🍵☕ Sure, here is the link: [https://ko-fi.com/drnicefellow](https://ko-fi.com/drnicefellow). Please add a note on which one you want me to drink?