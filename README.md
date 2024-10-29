# LoRA Fashion

Code related to captioning images (people and garments), training LoRA's, and stacking the LoRA's in ComfyUI.

## Overview

This repository contains:

- **Captioning Scripts**: Scripts to generate captions for images of people and garments.
- **Training Scripts**: Scripts to train LoRA models using the Ostris AI Toolkit.
- **ComfyUI Workflow**: A workflow JSON file for ComfyUI to stack and use the trained LoRA models.

## Compute Environment

All training and experiments were conducted on **Runpod** instances, utilizing a 48GB VRAM A40 GPU for efficient LoRA training and ComfyUI testing.

## Libraries & Resources
- **Training**: Ostris AI Toolkit ([GitHub](https://github.com/ostris/ai-toolkit))
- **ComfyUI**: [ComfyUI Repository](https://github.com/comfyanonymous/ComfyUI)
  - **ComfyUI Setup Guide for runpod**: [YouTube Tutorial](https://www.youtube.com/watch?v=WQiUqAdGIr4&t=889s)
  - **Using ComfyUI with FLUX DEV**: [Video Walkthrough](https://www.youtube.com/watch?v=txDFK-RcUq4&t=485s)
    - Note that I ended up using the default LoRA loader and had to put the LoRA scripts in the models/lora folder instead of dedicated xlabs folder to make it work.
  - **ComfyUI Manager**: [ComfyUI-Manager GitHub](https://github.com/ltdrdata/ComfyUI-Manager) (for managing ComfyUI plugins)
  - **Custom Nodes for LoRA Stacking**: [Comfyroll Custom Nodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) (for stacking LoRAs in ComfyUI)

## Contents
- **Captioning Scripts**: Scripts to generate descriptive captions for image data, focusing on garments and background context.
- **LoRA Training**: Code and configurations for training LoRAs on FLUX DEV, using the Ostris AI toolkit.
- **ComfyUI Workflow JSON**: Drag-and-drop configuration for ComfyUI, enabling the stacking of LoRAs and organized pipeline setup.