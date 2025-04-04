import pandas as pd
import os
from dotenv import load_dotenv
import pymupdf4llm
import os
from io import BytesIO
import base64
from openai import OpenAI
from PyPDF2 import PdfReader
import re
from PIL import Image
from spire.presentation import *
from spire.presentation.common import *
load_dotenv("credentials.env")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")

# Convert downloaded files to Markdown format
def get_image_dict2(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    image_dict = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{img_str}"
        }
    }
    return image_dict

def is_pdf_flattened(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        has_annotations = any(page.annotations for page in reader.pages)
        
        if has_annotations:
            return False
        else:
            return True
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return True

df = pd.read_csv('empty_scrape.csv')
model_name = "gpt-4o-mini"
client = OpenAI()

for index,row in df.iterrows():
    if '.pdf' in row['pdf_path']:
        try:
            df.at[index,'pdf_flattened'] = False
            #get folder path
            folder_path = '/'.join(row['pdf_path'].split('/')[:-1])
            
            #convert pdf to md
            md_text = pymupdf4llm.to_markdown(row['pdf_path'],
                write_images=True,
                image_path=folder_path,
                image_format="jpeg")
            
            md_output_path = row['pdf_path'].replace('.pdf','.md')
            with open(md_output_path, 'w', encoding='utf-8') as f:
                f.write(md_text)

            placeholders = re.findall(r'!\[\]\((.*?)\)', md_text)
                
            # Replace each placeholder with the LLM-generated description or "NA"
            for placeholder in placeholders:
                # Generate a prompt with the specific image path
                image_dict = get_image_dict2(Image.open(placeholder))
                prompt = """You are an AI assistant tasked with summarizing images 
                for use in a RAG (Retrieval-Augmented Generation) chatbot system.
                Your goal is to create concise, highly descriptive summaries of images 
                that contain information about indices, statistics, or other relevant data. 
                These summaries will be embedded and used to retrieve the raw images later.
                \n\nYou will be provided with an image to analyze. The image may come from studies, 
                articles, or presentation slides.\n\nFirst, carefully observe the entire image. 
                Assess whether it contains relevant information or if it's solely a logo, brand emblem, 
                or unrelated/random image.\n\nIf the image contains relevant information 
                (such as statistics, charts, graphs, or textual data about indices):\n\n
                1. Provide a concise summary that captures the main subject and context of the image.\n
                2. Include any distinct details that would be useful for accurate retrieval.\n
                3. Focus on describing the content rather than interpreting it.\n
                4. Ensure your summary is optimized for retrieval by including key terms and concepts present in the image.\n\n
                If the entire image consists solely of a logo, brand emblem, or unrelated/random images that do not contain useful information:\n\n
                1. Simply return 'NA' as your response.\n\nFormat your response as follows:\n- For relevant images, provide your summary within <summary> tags.\n- For irrelevant images, simply write 'NA' without any tags.\n\n
                Remember, your goal is to create summaries that will enable accurate retrieval of the raw images based on their content. 
                Be descriptive and precise, but avoid unnecessary interpretation or speculation."""
                prompt = prompt.replace("{{IMAGE}}", image_dict['image_url']['url'])
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            image_dict
                        ]
                    }
                ]
                
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages
                )

                image_summary = completion.choices[0].message.content
                print(image_summary)
                md_text = md_text.replace(f"![]({placeholder})", f"![]({placeholder})[{image_summary}]" if image_summary != "NA" else "")
            
            # Write the updated markdown content back to the file
            with open(str(row['pdf_path']).replace('.pdf','_final.md'), 'w', encoding='utf-8') as file:
                file.write(md_text)
            df.at[index,'md_path'] = str(row['pdf_path']).replace('.pdf','_final.md')
        except Exception as e:
            print(f"Error converting {row['pdf_path']} to markdown: {e}")
            df.at[index,'error'] = True
            continue

df.to_csv('empty_conv2md.csv',index=False)