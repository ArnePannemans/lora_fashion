"""
This script generates descriptive captions for images of people using the OpenAI API.
The goal is to create image - caption pairs that can be used for training LoRA models on Flux DEV.
"""

import os
import base64
import requests
from dotenv import load_dotenv
from PIL import Image
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def encode_image(image_path):
    """
    Encodes an image to base64 format.

    Args:
        image_path (str): Path to the image file.

    Returns:
        str: Base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def generate_caption(image_base64, api_key, trigger_word):
    """
    Generates a descriptive caption for the provided image using the OpenAI API.

    Args:
        image_base64 (str): Base64 encoded string of the image.
        api_key (str): OpenAI API key.
        trigger_word (str): Trigger word for the person (e.g., the folder name).

    Returns:
        str: Generated caption.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"""
        Generate a descriptive caption for the image provided.
        Focus on describing the person's pose, situation, clothing details (such as color, style, and fit), and any visible background elements.
        Use '{trigger_word}' as the identifier for the person without delving into the person's specific identity.
        For example, if '{trigger_word}' is a male with a beard and curly hair standing with arms at sides, dressed in a dark suit and white shirt, describe it as:
        '{trigger_word}, a male with a beard and curly hair standing with arms at sides, dressed in a dark suit and white shirt.'
        Separate all contextual and background details from the individual description associated with '{trigger_word}'.
    """
    
    payload = {
        "model": "gpt-4o", 
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                    },
                ]
            }
        ],
        "max_tokens": 100,
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        logger.error(f"API request failed with status {response.status_code}: {response.text}")
        return None

def main():
    image_folder = "model_A"
    api_key = os.getenv("OPENAI_API_KEY")

    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            logger.info(f"Processing image: {image_name}")
            
            try:
                image_base64 = encode_image(image_path)
                caption = generate_caption(image_base64, api_key, trigger_word=image_folder)
                if caption:
                    # Save the caption in a text file with the same name as the image
                    txt_output_path = os.path.join(image_folder, f"{os.path.splitext(image_name)[0]}.txt")
                    with open(txt_output_path, "w") as txt_file:
                        txt_file.write(caption)
                    logger.info(f"Generated caption saved to {txt_output_path}")
                else:
                    logger.warning(f"No caption generated for {image_name}")

            except Exception as e:
                logger.error(f"Error processing {image_name}: {e}")

    logger.info("Caption generation completed.")

if __name__ == "__main__":
    main()