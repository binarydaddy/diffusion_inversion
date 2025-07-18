import os  
import base64
from openai import AzureOpenAI
from typing import List

class PromptGenerator:
    def __init__(self):
        self.setup_env()

    def setup_env(self):
        endpoint = os.getenv("ENDPOINT_URL")
        deployment = os.getenv("DEPLOYMENT_NAME")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        # Validate required environment variables
        if not endpoint:
            raise ValueError("ENDPOINT_URL environment variable is required")
        if not deployment:
            raise ValueError("DEPLOYMENT_NAME environment variable is required")
        if not subscription_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")  
        
        client = AzureOpenAI(  
            azure_endpoint=endpoint,  
            api_key=subscription_key,
            api_version="2025-01-01-preview",
        )

        self.endpoint = endpoint
        self.deployment = deployment
        self.subscription_key = subscription_key
        self.client = client
        return

    def get_image_category_graphic_type_from_base64_string(self, base64_string, additional_info=None):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a professional artist specializing in visual effects (VFX) files and technologies, determining the category of the given VFX image. \
                        Example category of the VFX image is one of the following: \
                        - Fire \
                        - Smoke \
                        - Water \
                        - Explosion \
                        - Other \
                        Your task is to determine the category of the given VFX image into clear and precise English. \
                        Also, graphic type of the VFX image must be determined. \
                        Example graphic type of the VFX image is one of the following: \
                        - 2D \
                        - 3D \
                        Give answer in the following format: \
                            {'category': 'Fire','graphic_type': '2D'}"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_string}"
                        }
                    }
                ]
            },
        ] 
        message = chat_prompt
        
        completion = self.client.chat.completions.create(  
            model=self.deployment,
            messages=message,
            max_tokens=800,  
            temperature=1,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )

        return completion.choices[0].message.content
    
    def get_image_prompt_from_base64_string(self, base64_string, additional_info=None):
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a professional artist specializing in visual effects (VFX) files and technologies, writing diffusion prompt for given VFX image. \
                            Your task is to accurately describe content, structure, size, position, direction and background color of VFX image into clear and precise English. \
                            If additional information is provided, reflect the information in the description. Make sure to finish the description within 50 words, and format the answer in diffusion prompt format."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Additional information of the given VFX image is as follows: {additional_info}. Make sure to include this information."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_string}"
                        }
                    }
                ]
            },
        ] 
        message = chat_prompt
        
        completion = self.client.chat.completions.create(  
            model=self.deployment,
            messages=message,
            max_tokens=800,  
            temperature=1,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )

        return completion.choices[0].message.content

    def get_image_prompt(self, image_path, prompt_korean=None):
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode('ascii')
        return self.get_image_prompt_from_base64_string(encoded_image, prompt_korean)
    
    def get_image_prompt_from_PIL(self, image, additional_info, prompt_korean=None):
        import io
        # Convert PIL image to bytes in PNG format
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        buffer.seek(0)
        encoded_image = base64.b64encode(buffer.read()).decode('ascii')
        return self.get_image_prompt_from_base64_string(encoded_image, additional_info=additional_info)
    
    def get_video_prompt(self, prompt_list:List[str]=None, images_path:List[str]=None, prompt_korean=None):
        
        if prompt_list is None and images_path is None:
            raise ValueError("Either prompt_list or images_path must be provided")
        
        if prompt_list is None:
            prompt_list = []
            for image_path in images_path:
                prompt_list.append(self.get_image_prompt(image_path, prompt_korean))
        
        video_prompt = ""
        for i, prompt in enumerate(prompt_list):
            frame_prompt = f"Frame {i+1} of {len(prompt_list)}: {prompt}\n"
            video_prompt += frame_prompt

        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a professional artist specializing in VFX video technology. \
                            Your task is to accurately describe the content, movement (direction), and color of video contents in clear and precise English, based on the provided description of each frame. \
                            Make sure to combine the description of each frame into a single description, and finish the description within 50 words, single paragraph."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{video_prompt}"
                    }
                ]
            },
        ]

        message = chat_prompt

        completion = self.client.chat.completions.create(  
            model=self.deployment,
            messages=message,
            max_tokens=800,  
            temperature=1,  
            top_p=1,  
            frequency_penalty=0,  
            presence_penalty=0,
            stop=None,  
            stream=False
        )

        return completion.choices[0].message.content
    
if __name__ == "__main__":
    image_path = "test.png"

    from PIL import Image
    image = Image.open(image_path)

    pg = PromptGenerator()
    prompt = pg.get_image_prompt_from_PIL(image)
    print(prompt)