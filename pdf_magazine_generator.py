import json
import base64
import io
import os
from datetime import datetime
from PIL import Image
import google.generativeai as genai
import torch
import numpy as np
import requests

class PDFMagazineGenerator:
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
        """Retrieve saved API key"""
        config = self.load_config()
        return config.get("gemini_api_key", "")
    
    def save_api_key(self, api_key):
        """Store API key"""
        if api_key:
            config = self.load_config()
            config["gemini_api_key"] = api_key
            self.save_config(config)
    
    def call_ollama(self, prompt, image, model):
        """Call the Ollama API"""
        try:
            # convert image to base64
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
            raise Exception(f"Ollama call failed: {str(e)}")
    
    def call_openai(self, prompt, image, api_key, model, url):
        """Call the OpenAI API"""
        try:
            # convert image to base64
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
            raise Exception(f"OpenAI call failed: {str(e)}")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_type": (["Gemini", "Ollama", "OpenAI"], {"default": "Gemini"}),
                "design_style": ([
                    "Modern Minimalist", "Industrial", "Natural", "Nordic", "Japanese", 
                    "American Country", "French Classical", "Mediterranean", "Vintage", "Modern Chinese", "Custom Style"
                ], {"default": "Modern Minimalist"}),
                "custom_style": ("STRING", {"default": "", "placeholder": "Enter description when choosing custom style"}),
                "custom_scene": ("STRING", {"default": "", "placeholder": "Specify the floorplan scenario (e.g., classroom, conference room, restaurant). Leave blank for auto detection"}),
                "magazine_style": ([
                    "Fashion Magazine", "Architecture Magazine", "Home Living Magazine", "Designer Portfolio", "Real Estate Brochure"
                ], {"default": "Home Living Magazine"}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "json_data")
    OUTPUT_IS_LIST = (True, False)
    
    FUNCTION = "generate_pdf_magazine"
    CATEGORY = "pdf_magazine"
    
    def generate_pdf_magazine(self, llm_type, design_style, custom_style, custom_scene, magazine_style, image):
        try:
            # 
            config = self.load_config()
            
            # LLM
            if llm_type == "Gemini":
                api_key = config.get("gemini_api")
                model = config.get("gemini_model", "gemini-2.5-flash")
                if not api_key:
                    return (["Error: Gemini API key not configured, set it via the global LLM manager"], ": Gemini API Key")
                genai.configure(api_key=api_key)
                model_instance = genai.GenerativeModel(model)
            elif llm_type == "Ollama":
                model = config.get("ollama_model", "gemma2:12b")
                # Ollama
                model_instance = None
            elif llm_type == "OpenAI":
                api_key = config.get("openai_key")
                model = config.get("openai_model", "gpt-4o-mini")
                url = config.get("openai_url", "https://api.openai.com/v1")
                if not api_key:
                    return (["Error: OpenAI API key not configured, set it via the global LLM manager"], ": OpenAI API Key")
                # OpenAI
                model_instance = None
            else:
                return (["Error: Unsupported LLM type"], "Error: Unsupported LLM type")
            
            # 
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = image
            
            #  base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 
            if design_style == "Custom Style" and custom_style.strip():
                effective_style = custom_style.strip()
            else:
                effective_style = design_style
            
            # 
            scene_instruction = ""
            if custom_scene.strip():
                scene_instruction = f"Note that this floorplan is for '{custom_scene.strip()}' scene, "
            else:
                scene_instruction = "First determine the applicable scenario for this floorplan (e.g., residence, office, store) and then"
            
            # PDF/DM
            prompt = f"""
As a professional interior designer, {scene_instruction}analyze this floorplan and use "{effective_style}" as the primary design style and "{magazine_style}" layout style and output a JSON array suitable for an A4 PDF magazine/brochure.

Important: output a JSON array containing page objects. Format must be [{{"page_type": "cover", ...}}, {{"page_type": "contents", ...}}, ...]

JSON structure requirements:
1. Cover page (Cover):
   - page_type: "cover"
    - title: Main title (limit 12 words, e.g., "Dream Home Design")
    - subtitle: Subtitle (limit 10 words describing style features)
    - description: Cover description (limit 25 words)
   - image_prompt: English image prompt for cover (within 80 tokens, stylized interior design image)

2. Contents:
   - page_type: "contents"
   - title: "Design contents guide"
   - room_list: Room list array, each item includes:
     - room_name: Room name
     - page_number: Page number
       - highlight: Highlight (limit 10 words)
    - image_prompt: English prompt for overall design concept (within 80 tokens, exclude floorplan)

3. Style overview page (Style Overview):
   - page_type: "style_overview"
    - title: "{effective_style} Style Analysis"
    - style_description: Detailed style description (limit 80 words)
    - key_elements: Key elements array (3-4 items, each limit 15 words)
    - color_palette: Color palette explanation (limit 40 words)
    - material_guide: Material usage explanation (limit 40 words)
    - image_prompt: {effective_style} English prompt for showcasing the style (within 80 tokens)

4. Room detail pages:
   - page_type: "room_detail"
   - room_name: Room name
    - main_title: Room design title (limit 15 words)
    - design_concept: Design concept (limit 50 words)
    - functional_layout: Functional layout description (limit 40 words)
    - material_selection: Material selection suggestions (limit 40 words)
    - lighting_design: Lighting design description (limit 30 words)
    - furniture_suggestion: Furniture arrangement suggestions (limit 40 words)
    - color_scheme: Color scheme (limit 30 words)
    - special_features: Special design highlights (limit 30 words)
   - image_prompts: Array of English prompts for multiple angles of the room, including:
     - main_view: English prompt for main view (within 80 tokens)
     - detail_view: English prompt for detail view (within 80 tokens)
     - corner_view: English prompt for corner or alternate view (within 80 tokens)

5. Summary page (Summary):
   - page_type: "summary"
   - title: "Design summary and suggestions"
    - overall_summary: Overall design summary (limit 60 words)
    - budget_tips: Budget tips (limit 40 words)
    - timeline_suggestion: Construction timeline suggestion (limit 30 words)
    - maintenance_guide: Maintenance guide (limit 40 words)
   - contact_info: "For detailed consultation, please contact the professional design team"
   - image_prompt: English prompt for overall home feel after completion (within 80 tokens)

Important constraints:
- Analyze according to actual room layout of the floorplan
- Ensure all design suggestions fit "{effective_style}"
- English prompts should be specific for AI image generation; recommended aspect ratios 16:9 or 4:3
  - Text should be professional and easy to understand for {magazine_style} reading experience
  - Each page must have a corresponding image_prompt
  - Text length must strictly follow the specified limits

  Please output in JSON array format, starting with [ and ending with ], without extra text.
  Example format:
[
  {{"page_type": "cover", "title": "...", "subtitle": "...", "description": "...", "image_prompt": "..."}},
  {{"page_type": "contents", "title": "...", "room_list": [...], "image_prompt": "..."}},
  ...
]
"""
            
            # LLM
            if llm_type == "Gemini":
                response = model_instance.generate_content([prompt, pil_image])
                response_text = response.text.strip()
            elif llm_type == "Ollama":
                response_text = self.call_ollama(prompt, pil_image, model)
            elif llm_type == "OpenAI":
                response_text = self.call_openai(prompt, pil_image, api_key, model, url)
            
            # , markdown 
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:].strip()
            
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            # JSON
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx+1]
            
            print(f"100: {response_text[:100]}...")
            
            #  JSON
            try:
                magazine_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON: {e}")
                print(f": {response_text[:500]}...")
                return (["JSON"], f"JSON: {str(e)}")
            
            # magazine_data
            if not isinstance(magazine_data, list):
                return ([": ,"], ": ,")
            
            # 
            image_prompts = []
            for page in magazine_data:
                if isinstance(page, dict):
                    if "image_prompt" in page:
                        prompt = page["image_prompt"]
                        # 
                        if isinstance(prompt, str):
                            image_prompts.append(prompt)
                        elif isinstance(prompt, dict):
                            # ,
                            if "prompt" in prompt:
                                image_prompts.append(str(prompt["prompt"]))
                            else:
                                # 
                                for value in prompt.values():
                                    if isinstance(value, str) and value.strip():
                                        image_prompts.append(value)
                                        break
                                else:
                                    image_prompts.append("interior design, modern style")
                        else:
                            image_prompts.append(str(prompt) if prompt else "interior design")
                    elif "image_prompts" in page:
                        # 
                        room_prompts = page["image_prompts"]
                        if isinstance(room_prompts, dict):
                            for key in ["main_view", "detail_view", "corner_view"]:
                                prompt = room_prompts.get(key, "")
                                if isinstance(prompt, str) and prompt.strip():
                                    image_prompts.append(prompt)
                                elif prompt:
                                    image_prompts.append(str(prompt))
                        elif isinstance(room_prompts, list):
                            for prompt in room_prompts:
                                if isinstance(prompt, str):
                                    image_prompts.append(prompt)
                                elif isinstance(prompt, dict):
                                    # 
                                    for value in prompt.values():
                                        if isinstance(value, str) and value.strip():
                                            image_prompts.append(value)
                                            break
                                    else:
                                        image_prompts.append("interior design, modern style")
                                else:
                                    image_prompts.append(str(prompt) if prompt else "interior design")
            
            #  JSON 
            json_string = json.dumps(magazine_data, ensure_ascii=False, indent=2)
            
            return (image_prompts, json_string)
            
        except Exception as e:
            error_msg = f": {str(e)}"
            return ([error_msg], error_msg)


# PDF
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm, inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage, PageBreak, Table, TableStyle, BaseDocTemplate, PageTemplate, Frame
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfutils
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas
import tempfile

class PDFMagazineMaker:
    @classmethod
    def INPUT_TYPES(cls):
        # 
        font_folder = os.path.join(os.path.dirname(__file__), "fonts")
        font_files = []
        if os.path.exists(font_folder):
            font_files = [f for f in os.listdir(font_folder) if f.endswith('.ttf')]
        
        if not font_files:
            font_files = ["default"]
        
        return {
            "required": {
                "images": ("IMAGE",),
                "json_data": ("STRING",),
                "floorplanimage": ("IMAGE",),
                "template": (["", "", "", "", ""], {"default": ""}),
                "layout": (["A-", "B-", "C-", "D-"], {"default": "A-"}),
                "font": (font_files, {"default": font_files[0] if font_files else "default"}),
                "output_path": ("STRING", {"default": "./output/magazine.pdf"}),
            }
        }
    
    INPUT_IS_LIST = (True, False, False, False, False, False, False)  # (images, json_data, floorplanimage, template, layout, font, output_path)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    
    FUNCTION = "make_pdf_magazine"
    CATEGORY = "pdf_magazine"
    
    def __init__(self):
        # 
        self.templates = {
            "": {
                "title_color": colors.HexColor("#2C3E50"),
                "subtitle_color": colors.HexColor("#34495E"),
                "content_color": colors.HexColor("#2C3E50"),
                "accent_color": colors.HexColor("#3498DB"),
                "bg_color": colors.white
            },
            "": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#636E72"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#0984E3"),
                "bg_color": colors.white
            },
            "": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#6C5CE7"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#A29BFE"),
                "bg_color": colors.HexColor("#F8F9FA")
            },
            "": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#636E72"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#00B894"),
                "bg_color": colors.white
            },
            "": {
                "title_color": colors.HexColor("#8B4513"),
                "subtitle_color": colors.HexColor("#CD853F"),
                "content_color": colors.HexColor("#654321"),
                "accent_color": colors.HexColor("#DEB887"),
                "bg_color": colors.HexColor("#FFF8DC")
            }
        }
    
    def tensor_to_pil(self, tensor):
        """ tensor  PIL ,1088x960"""
        try:
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:
                    tensor = tensor.squeeze(0)
                elif tensor.dim() == 2:
                    # RGB
                    tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
                
                # tensor
                if tensor.max() <= 1.0:
                    # 0-1,0-255
                    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    # 0-255
                    image_np = tensor.cpu().numpy().astype(np.uint8)
                
                # RGB
                if image_np.shape[-1] != 3:
                    if image_np.shape[-1] == 1:
                        image_np = np.repeat(image_np, 3, axis=-1)
                    elif image_np.shape[-1] == 4:
                        image_np = image_np[:, :, :3]  # alpha
                
                pil_image = Image.fromarray(image_np, 'RGB')
                
                print(f": {pil_image.size}")
                return pil_image
                
            elif isinstance(tensor, Image.Image):
                return tensor.convert('RGB')
            elif isinstance(tensor, np.ndarray):
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                return Image.fromarray(tensor, 'RGB')
            else:
                print(f": {type(tensor)}")
                return Image.new('RGB', (1088, 960), color='lightgray')
                
        except Exception as e:
            print(f": {e}")
            print(f"Tensor: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")
            # 
            return Image.new('RGB', (1088, 960), color='lightgray')
    
    def resize_floorplan_for_layout(self, pil_image, max_width_mm, max_height_mm):
        ""","""
        try:
            width, height = pil_image.size
            
            # ,mm
            aspect_ratio = width / height
            target_aspect = max_width_mm / max_height_mm
            
            if aspect_ratio > target_aspect:
                # 
                final_width_mm = max_width_mm
                final_height_mm = max_width_mm / aspect_ratio
            else:
                # 
                final_height_mm = max_height_mm
                final_width_mm = max_height_mm * aspect_ratio
            
            # 
            max_pixel_size = 2000
            scale_factor = min(max_pixel_size / width, max_pixel_size / height, 1.0)
            
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            if scale_factor < 1.0:
                resized_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f": {width}x{height} -> {new_width}x{new_height}")
            else:
                resized_image = pil_image
                print(f": {width}x{height}")
            
            return resized_image, final_width_mm, final_height_mm
            
        except Exception as e:
            print(f": {e}")
            return pil_image, max_width_mm * 0.8, max_height_mm * 0.8

    def crop_image_for_layout(self, pil_image, target_ratio="16:9"):
        """ - ,"""
        try:
            width, height = pil_image.size
            current_ratio = width / height
            
            if target_ratio == "16:9":
                target_ratio_value = 16 / 9
            elif target_ratio == "4:3":
                target_ratio_value = 4 / 3
            elif target_ratio == "1:1":
                target_ratio_value = 1
            else:
                target_ratio_value = 16 / 9  # 
            
            # ,
            if abs(current_ratio - target_ratio_value) < 0.1:
                # ,
                return pil_image
            elif current_ratio > target_ratio_value:
                # ,
                new_width = int(height * target_ratio_value)
                left = (width - new_width) // 2
                right = left + new_width
                pil_image = pil_image.crop((left, 0, right, height))
                print(f": {width}x{height} -> {new_width}x{height}")
            elif current_ratio < target_ratio_value:
                # ,
                new_height = int(width / target_ratio_value)
                top = (height - new_height) // 2
                bottom = top + new_height
                pil_image = pil_image.crop((0, top, width, bottom))
                print(f": {width}x{height} -> {width}x{new_height}")
            
            return pil_image
            
        except Exception as e:
            print(f": {e}")
            return pil_image
    
    def register_font(self, font_name):
        """"""
        try:
            if font_name != "default":
                font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('CustomFont', font_path))
                    return 'CustomFont'
            return 'Helvetica'  # 
        except Exception as e:
            print(f": {e}")
            return 'Helvetica'
    
    def create_full_bleed_image(self, pil_image, width=210*mm, height=297*mm):
        """"""
        try:
            # A4
            img_width, img_height = pil_image.size
            target_ratio = width / height
            current_ratio = img_width / img_height
            
            if current_ratio > target_ratio:
                # ,
                new_height = img_height
                new_width = int(img_height * target_ratio)
                left = (img_width - new_width) // 2
                pil_image = pil_image.crop((left, 0, left + new_width, img_height))
            else:
                # ,
                new_width = img_width
                new_height = int(img_width / target_ratio)
                top = (img_height - new_height) // 2
                pil_image = pil_image.crop((0, top, img_width, top + new_height))
            
            # （DPI）
            target_pixels_w = int(width * 72 / 25.4)  # mmpixels (72 DPI)
            target_pixels_h = int(height * 72 / 25.4)
            pil_image = pil_image.resize((target_pixels_w, target_pixels_h), Image.Resampling.LANCZOS)
            
            return pil_image
        except Exception as e:
            print(f": {e}")
            return pil_image
    
    def wrap_text(self, text, max_width_chars, canvas_obj):
        """"""
        if not text:
            return [""]
        
        # :,
        lines = []
        text = str(text).strip()
        
        # ,
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # 
        while len(text) > max_width_chars:
            # （）
            cut_pos = max_width_chars
            for i in range(max_width_chars - 5, max_width_chars):
                if i < len(text) and text[i] in ',.!?;:,':
                    cut_pos = i + 1
                    break
            
            lines.append(text[:cut_pos].strip())
            text = text[cut_pos:].strip()
        
        if text:
            lines.append(text)
        
        return lines if lines else [""]
    
    
    def create_custom_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout, floorplan_image=None):
        """"""
        page_type = page_data.get("page_type", "")
        print(f"    : {page_type}")
        
        try:
            if page_type == "cover":
                print(f"    ...")
                self.draw_cover_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
            elif page_type == "contents":
                print(f"    ...")
                self.draw_contents_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout, floorplan_image)
            elif page_type == "style_overview":
                print(f"    ...")
                self.draw_style_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
            elif page_type == "room_detail":
                print(f"    ...")
                self.draw_room_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
            elif page_type == "summary":
                print(f"    ...")
                self.draw_summary_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
            else:
                print(f"    : {page_type}")
            print(f"     {page_type} ")
        except Exception as e:
            print(f"     {page_type} : {e}")
            # ,
            try:
                canvas_obj.setFillColor(colors.black)
                canvas_obj.setFont("Helvetica", 24)
                canvas_obj.drawCentredString(105*mm, 150*mm, f": {page_type}")
                canvas_obj.setFont("Helvetica", 12)
                canvas_obj.drawCentredString(105*mm, 130*mm, f": {str(e)[:50]}...")
                print(f"    ")
            except:
                print(f"    ")
    
    def draw_cover_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """ - """
        if layout == "A-":
            self.draw_cover_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "B-":
            self.draw_cover_layout_b(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "C-":
            self.draw_cover_layout_c(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "D-":
            self.draw_cover_layout_d(canvas_obj, page_data, images_for_page, template_config, font_name)
        else:
            # A
            self.draw_cover_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
    
    def draw_cover_layout_a(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """A:  - ,"""
        # 
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f": {e}")
                bg_img = Image.new('RGB', (int(210*mm*72/25.4), int(297*mm*72/25.4)), color='lightgray')
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
        
        # 
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.4)
        canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        # 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 42)
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 12, canvas_obj)
        y_start = 180*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*12*mm, line)
        
        # 
        canvas_obj.setFont(font_name, 20)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 20, canvas_obj)
        y_start = 130*mm
        for i, line in enumerate(subtitle_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*7*mm, line)
        
        # 
        canvas_obj.setFont(font_name, 14)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 28, canvas_obj)
        y_start = 90*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*5*mm, line)
    
    def draw_cover_layout_b(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """B:  - ,"""
        # （2/3）
        if images_for_page:
            try:
                img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(img, 0, 99*mm, width=210*mm, height=198*mm)
            except Exception as e:
                print(f": {e}")
        
        #  - 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.9)
        canvas_obj.rect(0, 0, 210*mm, 99*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 20*mm
        
        #  - 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 36)
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 15, canvas_obj)
        y_start = 75*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawString(margin, y_start - i*10*mm, line)
        
        # 
        canvas_obj.setFillColor(template_config["subtitle_color"])
        canvas_obj.setFont(font_name, 18)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 25, canvas_obj)
        y_start = 45*mm
        for i, line in enumerate(subtitle_lines):
            canvas_obj.drawString(margin, y_start - i*6*mm, line)
        
        # 
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 12)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 32, canvas_obj)
        y_start = 25*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawString(margin, y_start - i*4*mm, line)
    
    def draw_cover_layout_c(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """C:  - ,,"""
        # ,
        if not images_for_page:
            return
        
        # ,
        #  - 
        try:
            img1 = self.crop_image_for_layout(images_for_page[0], "4:3")
            canvas_obj.drawInlineImage(img1, 0, 170*mm, width=126*mm, height=127*mm)
            print(" ()")
        except Exception as e:
            print(f"1: {e}")
        
        #  - ,
        try:
            img_index = 1 if len(images_for_page) > 1 else 0
            img2 = self.crop_image_for_layout(images_for_page[img_index], "1:1")
            canvas_obj.drawInlineImage(img2, 126*mm, 220*mm, width=84*mm, height=77*mm)
            print(" ()")
        except Exception as e:
            print(f"2: {e}")
        
        #  - 
        try:
            img_index = 2 if len(images_for_page) > 2 else (0 if len(images_for_page) == 1 else 1)
            img3 = self.crop_image_for_layout(images_for_page[img_index], "16:9")
            canvas_obj.drawInlineImage(img3, 126*mm, 170*mm, width=84*mm, height=47*mm)
            print(" ()")
        except Exception as e:
            print(f"3: {e}")
        
        #  - ,
        title_area_height = 50*mm
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.95)
        canvas_obj.rect(0, 120*mm, 210*mm, title_area_height, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 15*mm
        
        #  - ,
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 36)  # 
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 12, canvas_obj)
        y_start = 155*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawString(margin, y_start - i*10*mm, line)
        
        #  - 
        canvas_obj.setFillColor(template_config["subtitle_color"])
        canvas_obj.setFont(font_name, 16)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 24, canvas_obj)
        y_start = 135*mm
        for i, line in enumerate(subtitle_lines):
            canvas_obj.drawString(margin, y_start - i*6*mm, line)
        
        #  - 
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.8)
        canvas_obj.rect(0, 0, 210*mm, 120*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        #  - 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 14)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 30, canvas_obj)
        y_start = 100*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawString(margin, y_start - i*5*mm, line)
    
    def draw_cover_layout_d(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """D:  - A（）"""
        # DA,
        self.draw_cover_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
    
    def draw_contents_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout, floorplan_image=None):
        """ - """
        
        # （）
        if images_for_page:
            try:
                # 
                bg_img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
                print("")
            except Exception as e:
                print(f": {e}")
        
        # （）
        if floorplan_image:
            try:
                print("...")
                
                # :mm
                max_width_mm = 180  # 
                max_height_mm = 120
                
                resized_floorplan, final_width_mm, final_height_mm = self.resize_floorplan_for_layout(
                    floorplan_image, max_width_mm, max_height_mm
                )
                
                # 
                x_offset = (210 - final_width_mm) / 2
                y_offset = 180 + (120 - final_height_mm) / 2
                
                canvas_obj.drawInlineImage(
                    resized_floorplan, 
                    x_offset*mm, 
                    y_offset*mm, 
                    width=final_width_mm*mm, 
                    height=final_height_mm*mm
                )
                print(f": {final_width_mm:.1f}x{final_height_mm:.1f}mm")
            except Exception as e:
                print(f": {e}")
        
        # 
        text_bg_color = template_config.get("bg_color", colors.white)
        canvas_obj.setFillColor(text_bg_color)
        canvas_obj.setFillAlpha(0.9)
        canvas_obj.rect(10*mm, 10*mm, 190*mm, 140*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 20*mm
        
        # 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 32)
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 130*mm, title)
        
        #  - ,
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 18)  # 
        
        room_list = page_data.get("room_list", [])
        col1_x = margin
        col2_x = 115*mm
        y_start = 110*mm
        line_spacing = 16*mm  # 
        
        for i, room in enumerate(room_list):
            x_pos = col1_x if i % 2 == 0 else col2_x
            y_pos = y_start - (i // 2) * line_spacing
            
            # 
            if y_pos > 25*mm:
                # 
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 18)
                room_text = f"• {room.get('room_name', '')}"
                canvas_obj.drawString(x_pos, y_pos, room_text)
                
                # 
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 14)
                highlight = room.get('highlight', '')
                if highlight:
                    highlight_lines = self.wrap_text(highlight, 18, canvas_obj)
                    for j, line in enumerate(highlight_lines[:2]):  # 2
                        canvas_obj.drawString(x_pos + 8*mm, y_pos - (j+1)*5*mm, line)
    
    def draw_room_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """ - """
        if layout == "A-":
            self.draw_room_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "B-":
            self.draw_room_layout_b(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "C-":
            self.draw_room_layout_c(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "D-":
            self.draw_room_layout_d(canvas_obj, page_data, images_for_page, template_config, font_name)
        else:
            # A
            self.draw_room_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
    
    def draw_room_layout_a(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """A:  - ,,"""
        # 3
        if len(images_for_page) >= 3:
            # 2/3
            try:
                main_img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(main_img, 0, 158*mm, width=210*mm, height=139*mm)
                
                # 
                left_img = self.crop_image_for_layout(images_for_page[1], "4:3")
                canvas_obj.drawInlineImage(left_img, 0, 79*mm, width=105*mm, height=79*mm)
                
                # 
                right_img = self.crop_image_for_layout(images_for_page[2], "4:3")
                canvas_obj.drawInlineImage(right_img, 105*mm, 79*mm, width=105*mm, height=79*mm)
                
            except Exception as e:
                print(f": {e}")
        
        #  - ,
        text_start_y = 70*mm
        margin = 10*mm
        col_width = 90*mm
        col1_x = margin
        col2_x = margin + col_width + 10*mm
        
        # 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 18)  # 
        title = page_data.get("main_title", "")
        canvas_obj.drawString(col1_x, text_start_y, title)
        
        #  - 
        sections = [
            ("", page_data.get("design_concept", "")),
            ("", page_data.get("functional_layout", "")),
            ("", page_data.get("material_selection", "")),
            ("", page_data.get("lighting_design", "")),
            ("", page_data.get("furniture_suggestion", "")),
            ("", page_data.get("color_scheme", "")),
            ("", page_data.get("special_features", ""))
        ]
        
        #  - 4
        y_pos = text_start_y - 8*mm
        for i, (section_title, section_content) in enumerate(sections[:4]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 10)  # 
                canvas_obj.drawString(col1_x, y_pos, section_title)
                y_pos -= 3.5*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 9)  # 
                content_lines = self.wrap_text(section_content, 20, canvas_obj)
                for line in content_lines[:2]:  # 2
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col1_x, y_pos, line)
                        y_pos -= 3*mm
                y_pos -= 2*mm  # 
        
        #  - 3
        y_pos = text_start_y - 8*mm
        for i, (section_title, section_content) in enumerate(sections[4:]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 10)  # 
                canvas_obj.drawString(col2_x, y_pos, section_title)
                y_pos -= 3.5*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 9)  # 
                content_lines = self.wrap_text(section_content, 20, canvas_obj)
                for line in content_lines[:2]:  # 2
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col2_x, y_pos, line)
                        y_pos -= 3*mm
                y_pos -= 2*mm  # 
    
    def draw_room_layout_b(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """B:  - ,"""
        # 
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f": {e}")
        
        # 
        title = page_data.get("main_title", "")
        design_concept = page_data.get("design_concept", "")
        
        #  - 
        title_lines = self.wrap_text(title, 12, canvas_obj)  # 
        concept_lines = self.wrap_text(design_concept, 16, canvas_obj)  # 
        
        # : +  +  +  + 
        title_height = len(title_lines) * 8*mm  # 
        concept_height = len(concept_lines[:3]) * 7*mm  # 
        required_height = title_height + 18*mm + 10*mm + concept_height + 30*mm  # 
        required_width = 130*mm  # 
        
        #  - 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.95)  # 
        canvas_obj.rect(25*mm, 297*mm-required_height-15*mm, required_width, required_height, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 35*mm  # 
        y_start = 297*mm - 30*mm  # 
        
        #  - 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 18)  # 
        for i, line in enumerate(title_lines):
            canvas_obj.drawString(margin, y_start - i*8*mm, line)  # 
        
        y_pos = y_start - len(title_lines)*8*mm - 18*mm  # 
        
        #  - 
        canvas_obj.setFillColor(template_config["accent_color"])
        canvas_obj.setFont(font_name, 13)  # 
        canvas_obj.drawString(margin, y_pos, "")
        y_pos -= 10*mm  # 
        
        #  - 
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 11)  # 
        for line in concept_lines[:3]:  # 3
            canvas_obj.drawString(margin, y_pos, line)
            y_pos -= 7*mm  # 
        
        #  - 
        sections = [
            ("", page_data.get("functional_layout", "")),
            ("", page_data.get("material_selection", "")),
            ("", page_data.get("lighting_design", "")),
            ("", page_data.get("furniture_suggestion", "")),
            ("", page_data.get("color_scheme", "")),
            ("", page_data.get("special_features", ""))
        ]
        
        # ,
        # 6,3
        left_sections = sections[:3]  # ,,
        right_sections = sections[3:]  # ,,
        
        # （）
        max_lines_left = 1  # ""
        for section_title, section_content in left_sections:
            if section_content:
                content_lines = self.wrap_text(section_content, 18, canvas_obj)  # 
                max_lines_left += 1 + len(content_lines[:2])  #  + （2）
        
        max_lines_right = 0
        for section_title, section_content in right_sections:
            if section_content:
                content_lines = self.wrap_text(section_content, 18, canvas_obj)
                max_lines_right += 1 + len(content_lines[:2])
        
        total_lines = max(max_lines_left, max_lines_right + 1)  # +1 for title
        box_height = max(total_lines * 6*mm + 50*mm, 120*mm)  # ,120mm
        box_width = 160*mm  # 
        
        #  - ,
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.85)  # 
        canvas_obj.rect(40*mm, 15*mm, box_width, box_height, fill=1, stroke=0)  # X
        canvas_obj.setFillAlpha(1)
        
        #  - 
        margin_right = 50*mm  # 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 16)  # 
        y_pos = 15*mm + box_height - 15*mm  # Y
        canvas_obj.drawString(margin_right, y_pos, "")
        y_pos -= 15*mm  # 
        
        #  - ,
        left_x = margin_right
        left_y = y_pos
        for section_title, section_content in left_sections:
            if section_content and left_y > 25*mm:  # 
                canvas_obj.setFont(font_name, 12)  # 
                canvas_obj.drawString(left_x, left_y, f"{section_title}:")
                left_y -= 7*mm  # 
                
                canvas_obj.setFont(font_name, 11)  # 
                content_lines = self.wrap_text(section_content, 16, canvas_obj)  # 
                for line in content_lines[:3]:  # 3
                    if left_y > 20*mm:  # 
                        canvas_obj.drawString(left_x, left_y, line)
                        left_y -= 6*mm  # 
                left_y -= 6*mm  # 
        
        #  - ,
        right_x = margin_right + 75*mm  # ,
        right_y = y_pos
        for section_title, section_content in right_sections:
            if section_content and right_y > 25*mm:  # 
                canvas_obj.setFont(font_name, 12)  # 
                canvas_obj.drawString(right_x, right_y, f"{section_title}:")
                right_y -= 7*mm  # 
                
                canvas_obj.setFont(font_name, 11)  # 
                content_lines = self.wrap_text(section_content, 16, canvas_obj)  # 
                for line in content_lines[:3]:  # 3
                    if right_y > 20*mm:  # 
                        canvas_obj.drawString(right_x, right_y, line)
                        right_y -= 6*mm  # 
                right_y -= 6*mm  # 
    
    def draw_room_layout_c(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """C:  - ,,"""
        # 
        if len(images_for_page) >= 1:
            try:
                main_img = self.crop_image_for_layout(images_for_page[0], "4:3")
                canvas_obj.drawInlineImage(main_img, 0, 150*mm, width=140*mm, height=147*mm)
            except Exception as e:
                print(f": {e}")
        
        # 
        if len(images_for_page) >= 2:
            try:
                small_img = self.crop_image_for_layout(images_for_page[1], "1:1")
                canvas_obj.drawInlineImage(small_img, 140*mm, 222*mm, width=70*mm, height=75*mm)
            except Exception as e:
                print(f": {e}")
        
        # 
        if len(images_for_page) >= 3:
            try:
                mid_img = self.crop_image_for_layout(images_for_page[2], "16:9")
                canvas_obj.drawInlineImage(mid_img, 140*mm, 150*mm, width=70*mm, height=72*mm)
            except Exception as e:
                print(f": {e}")
        
        # 
        if len(images_for_page) >= 1:
            try:
                bottom_img_index = 3 if len(images_for_page) > 3 else 0
                bottom_img = self.crop_image_for_layout(images_for_page[bottom_img_index], "16:9")
                canvas_obj.drawInlineImage(bottom_img, 0, 75*mm, width=210*mm, height=75*mm)
            except Exception as e:
                print(f": {e}")
        
        #  - 
        title = page_data.get("main_title", "")
        if title:
            title_lines = self.wrap_text(title, 12, canvas_obj)
            title_box_height = len(title_lines) * 5*mm + 10*mm
            title_box_width = 65*mm
            
            canvas_obj.setFillColor(colors.white)
            canvas_obj.setFillAlpha(0.9)
            canvas_obj.rect(210*mm-title_box_width-5*mm, 297*mm-title_box_height-5*mm, title_box_width, title_box_height, fill=1, stroke=0)
            canvas_obj.setFillAlpha(1)
            
            canvas_obj.setFillColor(template_config["title_color"])
            canvas_obj.setFont(font_name, 16)
            y_pos = 292*mm - 5*mm
            for line in title_lines:
                canvas_obj.drawString(210*mm-title_box_width, y_pos, line)
                y_pos -= 5*mm
        
        #  - （B）
        design_concept = page_data.get("design_concept", "")
        if design_concept:
            # 
            concept_lines = self.wrap_text(design_concept, 35, canvas_obj)
            concept_box_height = len(concept_lines) * 4*mm + 15*mm
            
            # 
            canvas_obj.setFillColor(colors.black)
            canvas_obj.setFillAlpha(0.75)
            canvas_obj.rect(10*mm, 200*mm, 190*mm, concept_box_height, fill=1, stroke=0)
            canvas_obj.setFillAlpha(1)
            
            # 
            canvas_obj.setFillColor(colors.white)
            canvas_obj.setFont(font_name, 14)
            canvas_obj.drawString(15*mm, 200*mm + concept_box_height - 8*mm, "")
            
            canvas_obj.setFont(font_name, 11)
            y_pos = 200*mm + concept_box_height - 15*mm
            for line in concept_lines[:4]:  # 4
                canvas_obj.drawString(15*mm, y_pos, line)
                y_pos -= 4*mm
        
        #  - （A）
        sections = [
            ("", page_data.get("functional_layout", "")),
            ("", page_data.get("material_selection", "")),
            ("", page_data.get("lighting_design", "")),
            ("", page_data.get("furniture_suggestion", "")),
            ("", page_data.get("color_scheme", "")),
            ("", page_data.get("special_features", ""))
        ]
        
        #  - 
        text_start_y = 65*mm
        margin = 10*mm
        col_width = 90*mm
        col1_x = margin
        col2_x = margin + col_width + 10*mm
        
        #  - 3
        y_pos = text_start_y
        for i, (section_title, section_content) in enumerate(sections[:3]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 11)
                canvas_obj.drawString(col1_x, y_pos, section_title)
                y_pos -= 4*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 10)
                content_lines = self.wrap_text(section_content, 22, canvas_obj)
                for line in content_lines[:2]:  # 2
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col1_x, y_pos, line)
                        y_pos -= 3.5*mm
                y_pos -= 3*mm  # 
        
        #  - 3
        y_pos = text_start_y
        for i, (section_title, section_content) in enumerate(sections[3:]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 11)
                canvas_obj.drawString(col2_x, y_pos, section_title)
                y_pos -= 4*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 10)
                content_lines = self.wrap_text(section_content, 22, canvas_obj)
                for line in content_lines[:2]:  # 2
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col2_x, y_pos, line)
                        y_pos -= 3.5*mm
                y_pos -= 3*mm  # 
    
    def draw_room_layout_d(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """D:  - ,"""
        
        # =====  () =====
        margin = 10*mm
        col_width = 90*mm  # 
        col1_x = margin
        col2_x = margin + col_width + 10*mm  # 
        
        #  - ,
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 20)  # 
        title = page_data.get("main_title", "")
        title_lines = self.wrap_text(title, 25, canvas_obj)  # 
        y_title = 280*mm
        for i, line in enumerate(title_lines[:2]):  # 2
            canvas_obj.drawString(col1_x, y_title - i*8*mm, line)
        
        # 
        title_offset = len(title_lines[:2]) * 8*mm
        
        #  - 
        sections = [
            ("", page_data.get("design_concept", "")),
            ("", page_data.get("functional_layout", "")),
            ("", page_data.get("material_selection", "")),
            ("", page_data.get("lighting_design", "")),
            ("", page_data.get("furniture_suggestion", "")),
            ("", page_data.get("color_scheme", "")),
            ("", page_data.get("special_features", ""))
        ]
        
        #  - 4,
        y_pos = 260*mm - title_offset  # ,
        for i, (section_title, section_content) in enumerate(sections[:4]):
            if section_content and y_pos > 150*mm:  # 
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 13)
                canvas_obj.drawString(col1_x, y_pos, section_title)
                y_pos -= 6*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 11)  # 
                content_lines = self.wrap_text(section_content, 18, canvas_obj)  # 
                for line in content_lines[:3]:  # 3
                    if y_pos > 150*mm:
                        canvas_obj.drawString(col1_x, y_pos, line)
                        y_pos -= 5*mm
                y_pos -= 4*mm  # 
        
        #  - 3,
        y_pos = 260*mm - title_offset  # ,
        for i, (section_title, section_content) in enumerate(sections[4:]):
            if section_content and y_pos > 150*mm:  # 
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 13)
                canvas_obj.drawString(col2_x, y_pos, section_title)
                y_pos -= 6*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 11)  # 
                content_lines = self.wrap_text(section_content, 18, canvas_obj)  # 
                for line in content_lines[:3]:  # 3
                    if y_pos > 150*mm:
                        canvas_obj.drawString(col2_x, y_pos, line)
                        y_pos -= 5*mm
                y_pos -= 4*mm  # 
        
        # =====  (,148mm) =====
        if len(images_for_page) >= 3:
            try:
                #  - 
                main_img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(main_img, 0, 79*mm, width=210*mm, height=69*mm)
                
                #  - 
                left_img = self.crop_image_for_layout(images_for_page[1], "4:3")
                canvas_obj.drawInlineImage(left_img, 0, 0, width=105*mm, height=79*mm)
                
                right_img = self.crop_image_for_layout(images_for_page[2], "4:3")
                canvas_obj.drawInlineImage(right_img, 105*mm, 0, width=105*mm, height=79*mm)
                
                print("D: (150-297mm),(0-148mm)")
                
            except Exception as e:
                print(f"D: {e}")
    
    def draw_style_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """ - """
        # 
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f": {e}")
        
        # 
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.5)
        canvas_obj.rect(0, 0, 210*mm, 148*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 15*mm
        
        # 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 32)
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 115*mm, title)
        
        # 
        canvas_obj.setFont(font_name, 14)  # 
        description = page_data.get("style_description", "")
        if description:
            desc_lines = self.wrap_text(description, 28, canvas_obj)
            y_pos = 95*mm
            for line in desc_lines[:4]:  # 4
                canvas_obj.drawString(margin, y_pos, line)
                y_pos -= 5*mm
        
        #  - 
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 14)
        canvas_obj.drawString(margin, 65*mm, "")
        
        canvas_obj.setFont(font_name, 12)  # 
        key_elements = page_data.get("key_elements", [])
        col1_x = margin
        col2_x = 110*mm
        y_start = 55*mm
        
        for i, element in enumerate(key_elements[:6]):  # 6
            x_pos = col1_x if i % 2 == 0 else col2_x
            y_pos = y_start - (i // 2) * 6*mm
            if y_pos > 15*mm:
                canvas_obj.drawString(x_pos, y_pos, f"• {element}")
    
    def draw_summary_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """ - """
        # 
        if images_for_page:
            try:
                img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(img, 0, 148*mm, width=210*mm, height=149*mm)
            except Exception as e:
                print(f": {e}")
        
        margin = 15*mm
        
        # 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 26)  # 
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 125*mm, title)
        
        #  - 
        sections = [
            ("", page_data.get("overall_summary", "")),
            ("", page_data.get("budget_tips", "")),
            ("", page_data.get("timeline_suggestion", "")),
            ("", page_data.get("maintenance_guide", ""))
        ]
        
        col1_x = margin
        col2_x = 110*mm
        y_start = 105*mm
        
        for i, (section_title, section_content) in enumerate(sections):
            if section_content:
                x_pos = col1_x if i % 2 == 0 else col2_x
                y_pos = y_start - (i // 2) * 25*mm
                
                if y_pos > 20*mm:
                    canvas_obj.setFillColor(template_config["accent_color"])
                    canvas_obj.setFont(font_name, 14)  # 
                    canvas_obj.drawString(x_pos, y_pos, section_title)
                    
                    canvas_obj.setFillColor(template_config["content_color"])
                    canvas_obj.setFont(font_name, 11)  # 
                    content_lines = self.wrap_text(section_content, 20, canvas_obj)
                    for j, line in enumerate(content_lines[:4]):  # 4
                        canvas_obj.drawString(x_pos, y_pos - (j+1)*4*mm, line)
        
        # 
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 16)  # 
        contact_info = page_data.get("contact_info", "")
        if contact_info:
            canvas_obj.drawCentredString(105*mm, 25*mm, contact_info)
    
    def make_pdf_magazine(self, images, json_data, floorplanimage, template, layout, font, output_path):
        try:
            # 
            print(f":")
            print(f"  images: {type(images)}")
            print(f"  json_data: {type(json_data)}")
            print(f"  floorplanimage: {type(floorplanimage)}")
            print(f"  template: {type(template)}")
            print(f"  font: {type(font)}")
            print(f"  output_path: {type(output_path)}")
            
            # ,
            if isinstance(json_data, list):
                json_data = json_data[0] if json_data else "{}"
            if isinstance(floorplanimage, list):
                floorplanimage = floorplanimage[0] if floorplanimage else None
            if isinstance(template, list):
                template = template[0] if template else ""
            if isinstance(layout, list):
                layout = layout[0] if layout else "A-"
            if isinstance(font, list):
                font = font[0] if font else "default"
            
            # JSON
            try:
                if isinstance(json_data, str):
                    magazine_data = json.loads(json_data)
                else:
                    magazine_data = json_data
            except json.JSONDecodeError as e:
                return (f"JSON: {str(e)}",)
            
            # 
            if not isinstance(magazine_data, list):
                return (": ,",)
            
            # 
            if isinstance(output_path, list):
                output_path = output_path[0] if output_path else "./output/magazine.pdf"
            elif not isinstance(output_path, str):
                output_path = str(output_path)
            
            # 
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path_parts = os.path.splitext(output_path)
            output_path = f"{path_parts[0]}_{timestamp}{path_parts[1]}"
            
            # 
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                os.makedirs("./output", exist_ok=True)
                output_path = os.path.join("./output", os.path.basename(output_path))
            
            # 
            font_name = self.register_font(font)
            
            # 
            template_config = self.templates.get(template, self.templates[""])
            
            print(f"Total  {len(images)}  images to process")
            print(f"Total  {len(magazine_data)}  pages to create")
            
            # 
            processed_floorplan = None
            if floorplanimage is not None:
                try:
                    processed_floorplan = self.tensor_to_pil(floorplanimage)
                    print(f",: {processed_floorplan.size}")
                except Exception as e:
                    print(f": {e}")
            
            # 
            images_distribution = self.distribute_images_to_pages(magazine_data, images)
            
            # CanvasPDF
            c = canvas.Canvas(output_path, pagesize=A4)
            
            for i, page_data in enumerate(magazine_data):
                try:
                    print(f"Start processing page  {i+1} : {page_data.get('page_type', 'unknown')}")
                    
                    # Get images for this page
                    page_images = images_distribution.get(i, [])
                    print(f"   {len(page_images)} ")
                    
                    # PIL
                    print(f"  ...")
                    processed_images = []
                    for j, img in enumerate(page_images):
                        try:
                            pil_img = self.tensor_to_pil(img)
                            processed_images.append(pil_img)
                            print(f"     {j+1} ")
                        except Exception as e:
                            print(f"     {j+1} : {e}")
                    
                    # 
                    print(f"  Start drawing page...")
                    self.create_custom_page(c, page_data, processed_images, template_config, font_name, layout, processed_floorplan)
                    print(f"  page {i+1}  drawing completed")
                    
                    # Add new page（）
                    if i < len(magazine_data) - 1:
                        print(f"  Add new page...")
                        c.showPage()
                        print(f"  New page added")
                        
                except Exception as e:
                    print(f" {i+1}  page processing failed: {e}")
                    # 
                    if i < len(magazine_data) - 1:
                        c.showPage()
            
            # PDF
            c.save()
            
            result = f"PDF magazine generated successfully: {output_path}"
            print(result)
            return (result,)
            
        except Exception as e:
            error_msg = f"PDF generation error: {str(e)}"
            print(error_msg)
            return (error_msg,)
    
    def distribute_images_to_pages(self, magazine_data, images):
        ""","""
        distribution = {}
        image_index = 0
        
        if not images:
            return distribution
        
        def get_next_image():
            nonlocal image_index
            if not images:
                return None
            img = images[image_index % len(images)]  # 
            image_index += 1
            return img
        
        for i, page_data in enumerate(magazine_data):
            page_type = page_data.get("page_type", "")
            
            if page_type == "cover":
                # Cover uses 1 image
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "contents":
                # Contents page uses 1 image
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "style_overview":
                # Style overview uses 1 image
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "room_detail":
                # Room detail page uses 3 images (ensure page filled)
                page_images = []
                for _ in range(3):
                    img = get_next_image()
                    if img is not None:
                        page_images.append(img)
                if page_images:
                    distribution[i] = page_images
                
            elif page_type == "summary":
                # Summary page uses 1 image
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
            
            else:
                # Other page types use 1 image
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
        
        return distribution


NODE_CLASS_MAPPINGS = {
    "PDFMagazineGenerator": PDFMagazineGenerator,
    "PDFMagazineMaker": PDFMagazineMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDFMagazineGenerator": "PDF File Generator",
    "PDFMagazineMaker": "PDF File Maker"
}
