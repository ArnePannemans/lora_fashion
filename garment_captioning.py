"""
This script generates descriptive captions for images of garments using the OpenAI API.
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
        trigger_word (str): Trigger word for the garment or person (e.g., the folder name).

    Returns:
        str: Generated caption.
    """
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    prompt = f"""
        Generate a descriptive caption for the image provided.
        Describe the garment identified by '{trigger_word}' with broad terms such as "sweater," "shirt," or "jacket" and simple colors.
        Avoid specific details about the garment's style or intricate features, as these are part of the garment's identity associated with '{trigger_word}'.
        If the garment is worn by a person, describe the person's pose, sex, race, body orientation, and any other clothing items or accessories separately from '{trigger_word}'.
        Additionally, include details about the background, lighting, and any other contextual elements that are not part of the garment's identity.
        
        For example, if the garment '{trigger_word}' is an orange sweater worn by a person standing outdoors, you might write:
        'A person wearing an orange {trigger_word} sweater, standing with hands in pockets, in an outdoor setting with trees and sunlight in the background.'
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
    image_folder = "SW_A"
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
