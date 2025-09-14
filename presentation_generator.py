import json
import base64
import io
import os
from PIL import Image
import google.generativeai as genai
import torch
import numpy as np
import requests

class PresentationGenerator:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "config.json")
    
    def load_config(self):
        """Load configuration file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_config(self, config):
        """Save configuration file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save config: {e}")
    
    def get_saved_api_key(self):
        """Get stored API key"""
        config = self.load_config()
        return config.get("gemini_api_key", "")
    
    def save_api_key(self, api_key):
        """Save API key"""
        if api_key:
            config = self.load_config()
            config["gemini_api_key"] = api_key
            self.save_config(config)
    
    def call_ollama(self, prompt, image, model):
        """Call the Ollama API"""
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False
            }
            response = requests.post(url, json=data, timeout=180)
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Ollama request failed: {str(e)}")
    
    def call_openai(self, prompt, image, api_key, model, url):
        """Call the OpenAI API"""
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4000
            }
            
            response = requests.post(f"{url}/chat/completions", 
                                   headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"OpenAI API error: {response.status_code}")
        except Exception as e:
            raise Exception(f"OpenAI request failed: {str(e)}")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_type": (["Gemini", "Ollama", "OpenAI"], {"default": "Gemini"}),
                "design_style": ([
                    "Modern Minimalist", "Industrial", "Natural", "Nordic", "Japanese",
                    "American Country", "French Classical", "Mediterranean", "Retro", "New Chinese", "Custom Style"
                ], {"default": "Modern Minimalist"}),
                "custom_style": ("STRING", {"default": "", "placeholder": "Enter style description when \"Custom Style\" is selected"}),
                "custom_scene": ("STRING", {"default": "", "placeholder": "Specify the scene for the floor plan (e.g., classroom, meeting room, restaurant); leave blank for auto detection"}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "json_data")
    OUTPUT_IS_LIST = (True, False)  # Only the first output is a list
    
    FUNCTION = "generate_presentation"
    CATEGORY = "presentation"
    
    def generate_presentation(self, llm_type, design_style, custom_style, custom_scene, image):
        try:
            # Load configuration
            config = self.load_config()

            # Check availability of the selected LLM
            if llm_type == "Gemini":
                api_key = config.get("gemini_api")
                model = config.get("gemini_model", "gemini-2.5-flash")
                if not api_key:
                    return (["Error: Gemini API key not configured; please configure it in the Global LLM Manager"], "Error: Gemini API key not configured")
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model)
            elif llm_type == "Ollama":
                model = config.get("ollama_model", "gemma2:12b")
                # Ollama model instantiation is handled during the request
                model_instance = None
            elif llm_type == "OpenAI":
                api_key = config.get("openai_key")
                model = config.get("openai_model", "gpt-4o-mini")
                url = config.get("openai_url", "https://api.openai.com/v1")
                if not api_key:
                    return (["Error: OpenAI API key not configured; please configure it in the Global LLM Manager"], "Error: OpenAI API key not configured")
                # OpenAI model instantiation is handled during the request
                model_instance = None
            else:
                return (["Error: Unsupported LLM type"], "Error: Unsupported LLM type")
            
            # Convert image format
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = image
            
            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Handle style selection
            if design_style == "Custom Style" and custom_style.strip():
                effective_style = custom_style.strip()
            else:
                effective_style = design_style
            
            # Handle scene specification
            scene_instruction = ""
            if custom_scene.strip():
                scene_instruction = f"Note that this floor plan represents a {custom_scene.strip()} scene, "
            else:
                scene_instruction = "First determine the applicable scene for this floor plan (e.g., residence, office, store), then"
            
            # Optimized prompt including second page format and decoration style
            prompt = f"""
Act as a professional interior designer. {scene_instruction}carefully analyze this floor plan and use "{effective_style}" as the main design style. Output a complete JSON file for creating an interior design proposal presentation.

JSON structure requirements:
1. First page (Title page):
   - value: "Interior Design Proposal Reference"
   - description: Briefly describe the overall design style (within 15 words, e.g., "Modern minimalist home")
   - image: English style prompt for a representative room (within 80 tokens)

2. Second page (Style overview):
   - value: "Style Overview"
   - description: "{effective_style} style features description"
   - topic1: "Circulation planning"
   - summary1: Flow planning analysis for this floor plan (max 35 words)
   - topic2: "Design concept"
   - summary2: Core design concept of {effective_style} (max 35 words)
   - topic3: "Color and material concept"
   - summary3: Color palette and material choices for {effective_style} (max 35 words)
   - topic4: "Spatial function planning"
   - summary4: Functional layout based on the floor plan (max 35 words)
   - image: English prompt describing the overall space in {effective_style} (within 80 tokens)

3. Middle pages (individual rooms):
   - value: English name of the room type (e.g., "Living Room", "Master Bedroom")
   - description: Detailed description of design style including color, materials, and layout suggestions
   - image: Corresponding English ComfyUI prompt (within 80 tokens)

4. Final page (Closing page):
   - value: "Thank you for viewing this presentation"
   - description: Style summary and core value (within 15 words, e.g., "Create a comfortable living space")
   - image: English prompt conveying the overall spatial feeling (within 80 tokens)

Important constraints:
- Descriptions on the first and last pages must be within 15 words
- The four summaries on the second page must each be within 35 words
- Analyze according to the actual room configuration and size ratio in the floor plan
- Ensure all rooms share the unified style "{effective_style}"
- English prompts must be precise and specific, suitable for AI image generation
- Descriptions should be professional and easy to understand with practical renovation advice

Output in pure JSON format with no additional text.
"""
            
            # Send request according to LLM type
            if llm_type == "Gemini":
                response = model_instance.generate_content([prompt, pil_image])
                response_text = response.text.strip()
            elif llm_type == "Ollama":
                response_text = self.call_ollama(prompt, pil_image, model)
            elif llm_type == "OpenAI":
                response_text = self.call_openai(prompt, pil_image, api_key, model, url)
            
            # Clean response text by removing possible markdown markers
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            presentation_data = json.loads(response_text)
            
            # Extract image prompts list
            image_prompts = []
            for item in presentation_data:
                if isinstance(item, dict):
                    prompt = item.get("image", "")
                    # Ensure the prompt is a string
                    if isinstance(prompt, str):
                        image_prompts.append(prompt)
                    elif isinstance(prompt, dict):
                        # If it's a dict, try extracting a valid string value
                        if "prompt" in prompt:
                            image_prompts.append(str(prompt["prompt"]))
                        else:
                            # Take the first string value
                            for value in prompt.values():
                                if isinstance(value, str) and value.strip():
                                    image_prompts.append(value)
                                    break
                            else:
                                image_prompts.append("interior design, modern style")
                    else:
                        image_prompts.append(str(prompt) if prompt else "interior design")
                else:
                    image_prompts.append("interior design")
            
            # Return the full JSON string
            json_string = json.dumps(presentation_data, ensure_ascii=False, indent=2)
            
            return (image_prompts, json_string)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            return ([error_msg], error_msg)

NODE_CLASS_MAPPINGS = {
    "PresentationGenerator": PresentationGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresentationGenerator": "Presentation Image Generator"
}