from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage
from config import base,dev,prod
from django.conf import settings
from pathlib import Path
from io import BytesIO
from PIL import Image
from io import BytesIO
from PIL import Image
import pandas as pd
import base64
import json
import requests
import os, sys
import base64
import json
import time
import re,io
import re
import ast

def convert_to_base64(file_path):
    """
    Convert PIL images to Base64 encoded strings
    :param file_path: Django FieldFile
    :return: Re-sized Base64 string
    """
    abs_file_path = Path(settings.MEDIA_ROOT) / Path(file_path.path)
    print(f"Absolute file path: {abs_file_path}")

    if abs_file_path.exists():
        pil_image = Image.open(abs_file_path)

        # Convert the image to RGB mode if it's in RGBA mode
        if pil_image.mode == 'RGBA':
            pil_image = pil_image.convert('RGB')
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")  # You can change the format if needed
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str
    else:
        print(f"File not found: {abs_file_path}")

def ollama_call_image_to_text(image_path,prompt,topk=0.9,topp=0.7,temperature=0.5):
    img_b64 = convert_to_base64(file_path=image_path)
    prompt = (
        """
            You're name is Rafiki, you have been created by NileAGI to help people with understanding 
            images and help them to search through images in the Rafiki search engine. Make sure
            precise and correct as your an expert in image analysis and understanding be helpful.

            Below is the instruction.
            Instuction: 
            """
        + prompt
    )

    image_part = {
    "type": "image_url",
    "image_url": f"data:image/jpeg;base64,{img_b64}",
    }

    text_part = {"type": "text", "text": prompt}
    content_parts = [image_part, text_part]
    human_messages = [HumanMessage(content=content_parts)]
    llm = ChatOllama(model="llava", temperature=temperature, top_k=topk, top_p=topp)
    output_parser = StrOutputParser()
    llm_output = llm.invoke(human_messages)
    parsed_output = output_parser.invoke(llm_output)
    return parsed_output

