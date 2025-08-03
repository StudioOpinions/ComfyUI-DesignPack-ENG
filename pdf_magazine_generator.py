import json
import base64
import io
import os
from PIL import Image
import google.generativeai as genai
import torch
import numpy as np

class PDFMagazineGenerator:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(__file__), "config.json")
    
    def load_config(self):
        """載入配置檔案"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_config(self, config):
        """儲存配置檔案"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"儲存配置失敗: {e}")
    
    def get_saved_api_key(self):
        """獲取已儲存的API Key"""
        config = self.load_config()
        return config.get("gemini_api_key", "")
    
    def save_api_key(self, api_key):
        """儲存API Key"""
        if api_key:
            config = self.load_config()
            config["gemini_api_key"] = api_key
            self.save_config(config)

    @classmethod
    def INPUT_TYPES(cls):
        # 嘗試載入已儲存的API Key
        instance = cls()
        saved_key = instance.get_saved_api_key()
        
        return {
            "required": {
                "api_key": ("STRING", {"default": saved_key, "placeholder": "輸入Gemini API Key (會自動儲存)"}),
                "model": (["gemini-2.5-pro", "gemini-2.5-flash"], {"default": "gemini-2.5-flash"}),
                "design_style": ([
                    "現代簡約風", "工業風", "自然風", "北歐風", "日式和風", 
                    "美式鄉村風", "法式古典風", "地中海風", "復古風", "新中式風"
                ], {"default": "現代簡約風"}),
                "magazine_style": ([
                    "時尚雜誌風", "建築專業雜誌", "居家生活雜誌", "設計師作品集", "房地產DM"
                ], {"default": "居家生活雜誌"}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "json_data")
    OUTPUT_IS_LIST = (True, False)
    
    FUNCTION = "generate_pdf_magazine"
    CATEGORY = "pdf_magazine"
    
    def generate_pdf_magazine(self, api_key, model, design_style, magazine_style, image):
        try:
            # 儲存API Key以供下次使用
            if api_key and api_key != self.get_saved_api_key():
                self.save_api_key(api_key)
                print("API Key已儲存")
            
            # 如果沒有輸入API Key，嘗試使用儲存的Key
            if not api_key:
                api_key = self.get_saved_api_key()
                if not api_key:
                    return (["錯誤: 請輸入Gemini API Key"], "錯誤: 請輸入Gemini API Key")
            
            # 配置 Gemini API
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            
            # 轉換圖片格式
            if isinstance(image, torch.Tensor):
                if image.dim() == 4:
                    image = image.squeeze(0)
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
            else:
                pil_image = image
            
            # 轉換為 base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # 為PDF雜誌/DM設計的提示詞
            prompt = f"""
請您作為專業室內設計師，分析此平面圖並以「{design_style}」為主要設計風格，以「{magazine_style}」的版面風格，輸出適合製作A4 PDF雜誌/DM的JSON陣列資料。

重要：請輸出一個JSON陣列，包含多個頁面物件。格式必須是：[{{"page_type": "cover", ...}}, {{"page_type": "contents", ...}}, ...]

JSON結構要求：
1. 封面頁（Cover）：
   - page_type: "cover"
   - title: 主標題（限制12個中文字，如"夢想居家設計"）
   - subtitle: 副標題（限制10個中文字，描述風格特色）
   - description: 封面說明文字（限制25個中文字）
   - image_prompt: 封面形象的英文提示詞（80個token內，適合雜誌封面）

2. 目錄頁（Contents）：
   - page_type: "contents"
   - title: "設計內容導覽"
   - room_list: 房間清單陣列，每個項目包含：
     - room_name: 房間中文名稱
     - page_number: 頁碼
     - highlight: 重點特色（限制10個中文字）
   - image_prompt: 整體平面圖概覽的英文提示詞（80個token內）

3. 風格介紹頁（Style Overview）：
   - page_type: "style_overview"
   - title: "{design_style}風格解析"
   - style_description: 風格詳細介紹（限制80個中文字）
   - key_elements: 風格關鍵元素陣列（3-4個項目，每項限制15個中文字）
   - color_palette: 色彩搭配說明（限制40個中文字）
   - material_guide: 材質運用說明（限制40個中文字）
   - image_prompt: {design_style}風格展示空間的英文提示詞（80個token內）

4. 各房間詳細頁面：
   - page_type: "room_detail"
   - room_name: 房間中文名稱
   - main_title: 房間設計標題（限制15個中文字）
   - design_concept: 設計理念（限制50個中文字）
   - functional_layout: 機能配置說明（限制40個中文字）
   - material_selection: 材質選擇建議（限制40個中文字）
   - lighting_design: 照明設計說明（限制30個中文字）
   - furniture_suggestion: 家具配置建議（限制40個中文字）
   - color_scheme: 色彩配置（限制30個中文字）
   - special_features: 特殊設計亮點（限制30個中文字）
   - image_prompts: 該房間的多角度英文提示詞陣列，包含：
     - main_view: 房間主要視角的英文提示詞（80個token內）
     - detail_view: 房間細節特寫的英文提示詞（80個token內）
     - corner_view: 房間角落或另一視角的英文提示詞（80個token內）

5. 總結頁（Summary）：
   - page_type: "summary"
   - title: "設計總結與建議"
   - overall_summary: 整體設計總結（限制60個中文字）
   - budget_tips: 預算建議（限制40個中文字）
   - timeline_suggestion: 施工時程建議（限制30個中文字）
   - maintenance_guide: 維護保養重點（限制40個中文字）
   - contact_info: "如需詳細諮詢，歡迎聯繫專業設計團隊"
   - image_prompt: 完成後整體居家感的英文提示詞（80個token內）

重要限制：
- 請根據平面圖實際房間配置進行分析
- 確保所有設計建議符合「{design_style}」風格
- 英文提示詞要精準具體，適合AI圖像生成，比例建議為16:9或4:3
- 中文內容要專業易懂，適合{magazine_style}的閱讀體驗
- 每個頁面都要有對應的image_prompt
- 文字長度務必嚴格控制在指定字數內

請直接輸出JSON陣列格式，開頭必須是 [ 結尾必須是 ]，不需要其他說明文字。
範例格式：
[
  {{"page_type": "cover", "title": "...", "subtitle": "...", "description": "...", "image_prompt": "..."}},
  {{"page_type": "contents", "title": "...", "room_list": [...], "image_prompt": "..."}},
  ...
]
"""
            
            # 發送請求給 Gemini
            response = model_instance.generate_content([prompt, pil_image])
            
            # 解析回應
            response_text = response.text.strip()
            
            # 清理回應文字，移除可能的 markdown 標記和其他格式
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:].strip()
            
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
            
            # 尋找JSON陣列的開始和結束
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                response_text = response_text[start_idx:end_idx+1]
            
            print(f"清理後的回應前100字元: {response_text[:100]}...")
            
            # 解析 JSON
            try:
                magazine_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"JSON解析錯誤: {e}")
                print(f"原始回應: {response_text[:500]}...")
                return (["JSON解析失敗"], f"JSON解析失敗: {str(e)}")
            
            # 確保magazine_data是列表格式
            if not isinstance(magazine_data, list):
                return (["錯誤: 回應格式不正確，需要陣列格式"], "錯誤: 回應格式不正確，需要陣列格式")
            
            # 提取圖片提示詞列表
            image_prompts = []
            for page in magazine_data:
                if isinstance(page, dict):
                    if "image_prompt" in page:
                        image_prompts.append(page["image_prompt"])
                    elif "image_prompts" in page:
                        # 房間詳細頁的多圖片提示詞
                        room_prompts = page["image_prompts"]
                        if isinstance(room_prompts, dict):
                            image_prompts.append(room_prompts.get("main_view", ""))
                            image_prompts.append(room_prompts.get("detail_view", ""))
                            image_prompts.append(room_prompts.get("corner_view", ""))
                        elif isinstance(room_prompts, list):
                            image_prompts.extend(room_prompts)
            
            # 返回完整的 JSON 字串
            json_string = json.dumps(magazine_data, ensure_ascii=False, indent=2)
            
            return (image_prompts, json_string)
            
        except Exception as e:
            error_msg = f"錯誤: {str(e)}"
            return ([error_msg], error_msg)


# PDF製作節點
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
        # 獲取字體文件列表
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
                "template": (["經典雜誌", "現代簡約", "時尚風格", "專業商務", "溫馨居家"], {"default": "經典雜誌"}),
                "layout": (["版型A-經典佈局", "版型B-滿版圖文", "版型C-藝術拼貼"], {"default": "版型A-經典佈局"}),
                "font": (font_files, {"default": font_files[0] if font_files else "default"}),
                "output_path": ("STRING", {"default": "./output/magazine.pdf"}),
            }
        }
    
    INPUT_IS_LIST = (True, False, False, False, False, False)  # (images, json_data, template, layout, font, output_path)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    
    FUNCTION = "make_pdf_magazine"
    CATEGORY = "pdf_magazine"
    
    def __init__(self):
        # 雜誌版型配置
        self.templates = {
            "經典雜誌": {
                "title_color": colors.HexColor("#2C3E50"),
                "subtitle_color": colors.HexColor("#34495E"),
                "content_color": colors.HexColor("#2C3E50"),
                "accent_color": colors.HexColor("#3498DB"),
                "bg_color": colors.white
            },
            "現代簡約": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#636E72"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#0984E3"),
                "bg_color": colors.white
            },
            "時尚風格": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#6C5CE7"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#A29BFE"),
                "bg_color": colors.HexColor("#F8F9FA")
            },
            "專業商務": {
                "title_color": colors.HexColor("#2D3436"),
                "subtitle_color": colors.HexColor("#636E72"),
                "content_color": colors.HexColor("#2D3436"),
                "accent_color": colors.HexColor("#00B894"),
                "bg_color": colors.white
            },
            "溫馨居家": {
                "title_color": colors.HexColor("#8B4513"),
                "subtitle_color": colors.HexColor("#CD853F"),
                "content_color": colors.HexColor("#654321"),
                "accent_color": colors.HexColor("#DEB887"),
                "bg_color": colors.HexColor("#FFF8DC")
            }
        }
    
    def tensor_to_pil(self, tensor):
        """將 tensor 轉換為 PIL 圖像，處理1088x960尺寸"""
        try:
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:
                    tensor = tensor.squeeze(0)
                elif tensor.dim() == 2:
                    # 灰度圖轉RGB
                    tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
                
                # 確保tensor在正確範圍內
                if tensor.max() <= 1.0:
                    # 0-1範圍，轉換為0-255
                    image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                else:
                    # 已經是0-255範圍
                    image_np = tensor.cpu().numpy().astype(np.uint8)
                
                # 確保是RGB格式
                if image_np.shape[-1] != 3:
                    if image_np.shape[-1] == 1:
                        image_np = np.repeat(image_np, 3, axis=-1)
                    elif image_np.shape[-1] == 4:
                        image_np = image_np[:, :, :3]  # 移除alpha通道
                
                pil_image = Image.fromarray(image_np, 'RGB')
                
                print(f"轉換後圖片尺寸: {pil_image.size}")
                return pil_image
                
            elif isinstance(tensor, Image.Image):
                return tensor.convert('RGB')
            elif isinstance(tensor, np.ndarray):
                if tensor.max() <= 1.0:
                    tensor = (tensor * 255).astype(np.uint8)
                return Image.fromarray(tensor, 'RGB')
            else:
                print(f"未知圖片類型: {type(tensor)}")
                return Image.new('RGB', (1088, 960), color='lightgray')
                
        except Exception as e:
            print(f"圖片轉換錯誤: {e}")
            print(f"Tensor形狀: {tensor.shape if hasattr(tensor, 'shape') else 'unknown'}")
            # 創建一個默認圖片
            return Image.new('RGB', (1088, 960), color='lightgray')
    
    def crop_image_for_layout(self, pil_image, target_ratio="16:9"):
        """根據版面需求裁切圖片"""
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
                target_ratio_value = 16 / 9  # 默認
            
            if current_ratio > target_ratio_value:
                # 圖片太寬，需要裁切寬度
                new_width = int(height * target_ratio_value)
                left = (width - new_width) // 2
                pil_image = pil_image.crop((left, 0, left + new_width, height))
            elif current_ratio < target_ratio_value:
                # 圖片太高，需要裁切高度
                new_height = int(width / target_ratio_value)
                top = (height - new_height) // 2
                pil_image = pil_image.crop((0, top, width, top + new_height))
            
            return pil_image
            
        except Exception as e:
            print(f"圖片裁切錯誤: {e}")
            return pil_image
    
    def register_font(self, font_name):
        """註冊字體"""
        try:
            if font_name != "default":
                font_path = os.path.join(os.path.dirname(__file__), "fonts", font_name)
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('CustomFont', font_path))
                    return 'CustomFont'
            return 'Helvetica'  # 默認字體
        except Exception as e:
            print(f"字體註冊失敗: {e}")
            return 'Helvetica'
    
    def create_full_bleed_image(self, pil_image, width=210*mm, height=297*mm):
        """創建滿版圖片"""
        try:
            # 調整圖片為滿版A4尺寸
            img_width, img_height = pil_image.size
            target_ratio = width / height
            current_ratio = img_width / img_height
            
            if current_ratio > target_ratio:
                # 圖片太寬，以高度為準
                new_height = img_height
                new_width = int(img_height * target_ratio)
                left = (img_width - new_width) // 2
                pil_image = pil_image.crop((left, 0, left + new_width, img_height))
            else:
                # 圖片太高，以寬度為準
                new_width = img_width
                new_height = int(img_width / target_ratio)
                top = (img_height - new_height) // 2
                pil_image = pil_image.crop((0, top, img_width, top + new_height))
            
            # 調整到精確尺寸（考慮DPI）
            target_pixels_w = int(width * 72 / 25.4)  # 轉換mm到pixels (72 DPI)
            target_pixels_h = int(height * 72 / 25.4)
            pil_image = pil_image.resize((target_pixels_w, target_pixels_h), Image.Resampling.LANCZOS)
            
            return pil_image
        except Exception as e:
            print(f"滿版圖片處理錯誤: {e}")
            return pil_image
    
    def wrap_text(self, text, max_width_chars, canvas_obj):
        """智能文字換行 - 優化中文處理"""
        if not text:
            return [""]
        
        # 中文字符按字數切分，英文按單詞切分
        lines = []
        current_line = ""
        
        # 處理強制換行符
        paragraphs = text.replace('\n', '\\n').split('\\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                if current_line:
                    lines.append(current_line)
                    current_line = ""
                continue
                
            i = 0
            while i < len(paragraph):
                char = paragraph[i]
                
                # 如果遇到標點符號，優先在此處換行
                if char in '，。！？；：':
                    current_line += char
                    if len(current_line) >= max_width_chars * 0.8:  # 80%長度時考慮換行
                        lines.append(current_line)
                        current_line = ""
                # 如果是中文字符或其他寬字符
                elif ord(char) > 127:
                    if len(current_line) >= max_width_chars:
                        lines.append(current_line)
                        current_line = char
                    else:
                        current_line += char
                else:
                    # 英文字符，以單詞為單位
                    word_end = i
                    while word_end < len(paragraph) and ord(paragraph[word_end]) <= 127 and paragraph[word_end] not in ' \t':
                        word_end += 1
                    
                    word = paragraph[i:word_end]
                    if len(current_line) + len(word) > max_width_chars and current_line:
                        lines.append(current_line)
                        current_line = word
                    else:
                        current_line += word
                    
                    i = word_end - 1
                
                i += 1
            
            # 段落結束，如果有內容就加入行列表
            if current_line:
                lines.append(current_line)
                current_line = ""
        
        return lines if lines else [""]
    
    
    def create_custom_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """創建自定義頁面"""
        page_type = page_data.get("page_type", "")
        
        if page_type == "cover":
            self.draw_cover_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
        elif page_type == "contents":
            self.draw_contents_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
        elif page_type == "style_overview":
            self.draw_style_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
        elif page_type == "room_detail":
            self.draw_room_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
        elif page_type == "summary":
            self.draw_summary_page(canvas_obj, page_data, images_for_page, template_config, font_name, layout)
    
    def draw_cover_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """繪製封面頁 - 支援三種版型"""
        if layout == "版型A-經典佈局":
            self.draw_cover_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "版型B-滿版圖文":
            self.draw_cover_layout_b(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "版型C-藝術拼貼":
            self.draw_cover_layout_c(canvas_obj, page_data, images_for_page, template_config, font_name)
        else:
            # 默認使用版型A
            self.draw_cover_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
    
    def draw_cover_layout_a(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型A: 經典佈局 - 滿版背景，中央文字"""
        # 滿版背景圖片
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f"封面圖片處理錯誤: {e}")
                bg_img = Image.new('RGB', (int(210*mm*72/25.4), int(297*mm*72/25.4)), color='lightgray')
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
        
        # 半透明遮罩
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.4)
        canvas_obj.rect(0, 0, 210*mm, 297*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        # 主標題
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 42)
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 15, canvas_obj)
        y_start = 180*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*12*mm, line)
        
        # 副標題
        canvas_obj.setFont(font_name, 20)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 25, canvas_obj)
        y_start = 130*mm
        for i, line in enumerate(subtitle_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*7*mm, line)
        
        # 描述文字
        canvas_obj.setFont(font_name, 14)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 35, canvas_obj)
        y_start = 90*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawCentredString(105*mm, y_start - i*5*mm, line)
    
    def draw_cover_layout_b(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型B: 滿版圖文 - 上下分割，文字區域透明底色"""
        # 上方大圖（2/3版面）
        if images_for_page:
            try:
                img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(img, 0, 99*mm, width=210*mm, height=198*mm)
            except Exception as e:
                print(f"封面圖片處理錯誤: {e}")
        
        # 下方文字區域 - 透明底色
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.9)
        canvas_obj.rect(0, 0, 210*mm, 99*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 20*mm
        
        # 主標題 - 左對齊
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 36)
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 18, canvas_obj)
        y_start = 75*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawString(margin, y_start - i*10*mm, line)
        
        # 副標題
        canvas_obj.setFillColor(template_config["subtitle_color"])
        canvas_obj.setFont(font_name, 18)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 30, canvas_obj)
        y_start = 45*mm
        for i, line in enumerate(subtitle_lines):
            canvas_obj.drawString(margin, y_start - i*6*mm, line)
        
        # 描述文字
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 12)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 40, canvas_obj)
        y_start = 25*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawString(margin, y_start - i*4*mm, line)
    
    def draw_cover_layout_c(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型C: 藝術拼貼 - 多圖片幾何排列"""
        # 左上大圖
        if len(images_for_page) >= 1:
            try:
                img1 = self.crop_image_for_layout(images_for_page[0], "4:3")
                canvas_obj.drawInlineImage(img1, 0, 148*mm, width=126*mm, height=149*mm)
            except Exception as e:
                print(f"圖片1處理錯誤: {e}")
        
        # 右上圖
        if len(images_for_page) >= 2:
            try:
                img2 = self.crop_image_for_layout(images_for_page[1] if len(images_for_page) > 1 else images_for_page[0], "1:1")
                canvas_obj.drawInlineImage(img2, 126*mm, 198*mm, width=84*mm, height=99*mm)
            except Exception as e:
                print(f"圖片2處理錯誤: {e}")
        
        # 右中圖
        if len(images_for_page) >= 3:
            try:
                img3 = self.crop_image_for_layout(images_for_page[2] if len(images_for_page) > 2 else images_for_page[0], "1:1")
                canvas_obj.drawInlineImage(img3, 126*mm, 148*mm, width=84*mm, height=50*mm)
            except Exception as e:
                print(f"圖片3處理錯誤: {e}")
        
        # 文字區域 - 透明黑色底
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.7)
        canvas_obj.rect(0, 0, 210*mm, 148*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 15*mm
        
        # 主標題 - 白色大字
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 40)
        title = page_data.get("title", "")
        title_lines = self.wrap_text(title, 12, canvas_obj)
        y_start = 120*mm
        for i, line in enumerate(title_lines):
            canvas_obj.drawString(margin, y_start - i*12*mm, line)
        
        # 副標題 - 對角線排列
        canvas_obj.setFont(font_name, 16)
        subtitle = page_data.get("subtitle", "")
        subtitle_lines = self.wrap_text(subtitle, 25, canvas_obj)
        y_start = 80*mm
        for i, line in enumerate(subtitle_lines):
            x_offset = margin + i * 5*mm  # 階梯式排列
            canvas_obj.drawString(x_offset, y_start - i*6*mm, line)
        
        # 描述文字
        canvas_obj.setFont(font_name, 12)
        description = page_data.get("description", "")
        desc_lines = self.wrap_text(description, 35, canvas_obj)
        y_start = 50*mm
        for i, line in enumerate(desc_lines):
            canvas_obj.drawString(margin, y_start - i*4*mm, line)
    
    def draw_contents_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """繪製目錄頁 - 滿版設計"""
        # 上半部大圖
        if images_for_page:
            try:
                img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(img, 0, 148.5*mm, width=210*mm, height=148.5*mm)
            except Exception as e:
                print(f"目錄頁圖片處理錯誤: {e}")
        
        # 下半部內容區域
        margin = 15*mm
        
        # 標題
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 30)  # 調大字體
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 125*mm, title)
        
        # 目錄內容 - 雙欄設計
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 16)  # 調大字體
        
        room_list = page_data.get("room_list", [])
        col1_x = margin
        col2_x = 110*mm
        y_start = 105*mm
        
        for i, room in enumerate(room_list):
            x_pos = col1_x if i % 2 == 0 else col2_x
            y_pos = y_start - (i // 2) * 12*mm
            
            # 確保不超出頁面
            if y_pos > 20*mm:
                room_text = f"• {room.get('room_name', '')}"
                canvas_obj.drawString(x_pos, y_pos, room_text)
                
                canvas_obj.setFont(font_name, 12)  # 調大字體
                highlight = room.get('highlight', '')
                if highlight:
                    highlight_lines = self.wrap_text(highlight, 20, canvas_obj)
                    for j, line in enumerate(highlight_lines[:2]):  # 最多2行
                        canvas_obj.drawString(x_pos + 5*mm, y_pos - (j+1)*4*mm, line)
                
                canvas_obj.setFont(font_name, 16)  # 調大字體
    
    def draw_room_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """繪製房間詳細頁 - 支援三種版型"""
        if layout == "版型A-經典佈局":
            self.draw_room_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "版型B-滿版圖文":
            self.draw_room_layout_b(canvas_obj, page_data, images_for_page, template_config, font_name)
        elif layout == "版型C-藝術拼貼":
            self.draw_room_layout_c(canvas_obj, page_data, images_for_page, template_config, font_name)
        else:
            # 默認使用版型A
            self.draw_room_layout_a(canvas_obj, page_data, images_for_page, template_config, font_name)
    
    def draw_room_layout_a(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型A: 經典佈局 - 上大圖，下雙小圖，底部文字"""
        # 確保有3張圖片可用
        if len(images_for_page) >= 3:
            # 大圖佔據上方2/3空間
            try:
                main_img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(main_img, 0, 158*mm, width=210*mm, height=139*mm)
                
                # 下方左圖
                left_img = self.crop_image_for_layout(images_for_page[1], "4:3")
                canvas_obj.drawInlineImage(left_img, 0, 79*mm, width=105*mm, height=79*mm)
                
                # 下方右圖
                right_img = self.crop_image_for_layout(images_for_page[2], "4:3")
                canvas_obj.drawInlineImage(right_img, 105*mm, 79*mm, width=105*mm, height=79*mm)
                
            except Exception as e:
                print(f"房間頁圖片處理錯誤: {e}")
        
        # 文字區域 - 下方空間，雙欄設計
        text_start_y = 70*mm
        margin = 10*mm
        col_width = 90*mm
        col1_x = margin
        col2_x = margin + col_width + 10*mm
        
        # 房間標題
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 18)  # 調大字體
        title = page_data.get("main_title", "")
        canvas_obj.drawString(col1_x, text_start_y, title)
        
        # 設計內容分兩欄
        sections = [
            ("設計理念", page_data.get("design_concept", "")),
            ("機能配置", page_data.get("functional_layout", "")),
            ("材質選擇", page_data.get("material_selection", "")),
            ("特殊亮點", page_data.get("special_features", ""))
        ]
        
        # 左欄
        y_pos = text_start_y - 8*mm
        for i, (section_title, section_content) in enumerate(sections[:2]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 11)  # 調大字體
                canvas_obj.drawString(col1_x, y_pos, section_title)
                y_pos -= 4*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 10)  # 調大字體
                content_lines = self.wrap_text(section_content, 25, canvas_obj)
                for line in content_lines[:3]:  # 最多3行
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col1_x, y_pos, line)
                        y_pos -= 3.5*mm
                y_pos -= 2*mm  # 段落間距
        
        # 右欄
        y_pos = text_start_y - 8*mm
        for i, (section_title, section_content) in enumerate(sections[2:]):
            if section_content and y_pos > 10*mm:
                canvas_obj.setFillColor(template_config["accent_color"])
                canvas_obj.setFont(font_name, 11)  # 調大字體
                canvas_obj.drawString(col2_x, y_pos, section_title)
                y_pos -= 4*mm
                
                canvas_obj.setFillColor(template_config["content_color"])
                canvas_obj.setFont(font_name, 10)  # 調大字體
                content_lines = self.wrap_text(section_content, 25, canvas_obj)
                for line in content_lines[:3]:  # 最多3行
                    if y_pos > 10*mm:
                        canvas_obj.drawString(col2_x, y_pos, line)
                        y_pos -= 3.5*mm
                y_pos -= 2*mm  # 段落間距
    
    def draw_room_layout_b(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型B: 滿版圖文 - 滿版背景圖，文字透明底色覆蓋"""
        # 滿版背景圖片
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f"滿版圖片處理錯誤: {e}")
        
        # 左上角透明文字區域
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.9)
        canvas_obj.rect(15*mm, 200*mm, 120*mm, 80*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 20*mm
        
        # 房間標題
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 20)  # 調大字體
        title = page_data.get("main_title", "")
        canvas_obj.drawString(margin, 265*mm, title)
        
        # 設計理念
        canvas_obj.setFillColor(template_config["accent_color"])
        canvas_obj.setFont(font_name, 12)  # 調大字體
        canvas_obj.drawString(margin, 250*mm, "設計理念")
        
        canvas_obj.setFillColor(template_config["content_color"])
        canvas_obj.setFont(font_name, 11)  # 調大字體
        design_concept = page_data.get("design_concept", "")
        concept_lines = self.wrap_text(design_concept, 30, canvas_obj)
        y_pos = 240*mm
        for line in concept_lines[:3]:
            canvas_obj.drawString(margin, y_pos, line)
            y_pos -= 4*mm
        
        # 右下角透明文字區域
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.8)
        canvas_obj.rect(75*mm, 20*mm, 120*mm, 100*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        # 其他設計信息 - 白色字體
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 12)
        canvas_obj.drawString(85*mm, 105*mm, "設計特色")
        
        sections = [
            ("機能配置", page_data.get("functional_layout", "")),
            ("材質選擇", page_data.get("material_selection", "")),
            ("特殊亮點", page_data.get("special_features", ""))
        ]
        
        y_pos = 95*mm
        for section_title, section_content in sections:
            if section_content and y_pos > 30*mm:
                canvas_obj.setFont(font_name, 10)
                canvas_obj.drawString(85*mm, y_pos, f"{section_title}:")
                y_pos -= 4*mm
                
                canvas_obj.setFont(font_name, 9)
                content_lines = self.wrap_text(section_content, 35, canvas_obj)
                for line in content_lines[:2]:  # 最多2行
                    if y_pos > 30*mm:
                        canvas_obj.drawString(85*mm, y_pos, line)
                        y_pos -= 3.5*mm
                y_pos -= 3*mm  # 段落間距
    
    def draw_room_layout_c(self, canvas_obj, page_data, images_for_page, template_config, font_name):
        """版型C: 藝術拼貼 - 不規則圖片排列，文字穿插"""
        # 左側大圖
        if len(images_for_page) >= 1:
            try:
                main_img = self.crop_image_for_layout(images_for_page[0], "4:3")
                canvas_obj.drawInlineImage(main_img, 0, 150*mm, width=140*mm, height=147*mm)
            except Exception as e:
                print(f"主圖處理錯誤: {e}")
        
        # 右上小圖
        if len(images_for_page) >= 2:
            try:
                small_img = self.crop_image_for_layout(images_for_page[1], "1:1")
                canvas_obj.drawInlineImage(small_img, 140*mm, 222*mm, width=70*mm, height=75*mm)
            except Exception as e:
                print(f"小圖處理錯誤: {e}")
        
        # 右中圖
        if len(images_for_page) >= 3:
            try:
                mid_img = self.crop_image_for_layout(images_for_page[2], "16:9")
                canvas_obj.drawInlineImage(mid_img, 140*mm, 150*mm, width=70*mm, height=72*mm)
            except Exception as e:
                print(f"中圖處理錯誤: {e}")
        
        # 底部滿寬圖
        if len(images_for_page) >= 1:
            try:
                bottom_img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(bottom_img, 0, 75*mm, width=210*mm, height=75*mm)
            except Exception as e:
                print(f"底圖處理錯誤: {e}")
        
        # 文字區域 - 交錯排列
        # 右側文字區域1
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFillAlpha(0.95)
        canvas_obj.rect(145*mm, 190*mm, 60*mm, 25*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 16)  # 調大字體
        title = page_data.get("main_title", "")
        title_lines = self.wrap_text(title, 12, canvas_obj)
        canvas_obj.drawString(150*mm, 205*mm, title_lines[0] if title_lines else "")
        
        # 底部文字區域
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.85)
        canvas_obj.rect(0, 0, 210*mm, 75*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 15*mm
        
        # 設計內容 - 三欄排列
        sections = [
            ("設計理念", page_data.get("design_concept", "")),
            ("材質選擇", page_data.get("material_selection", "")),
            ("特殊亮點", page_data.get("special_features", ""))
        ]
        
        col_width = 60*mm
        for i, (section_title, section_content) in enumerate(sections):
            if section_content:
                x_pos = margin + i * col_width
                
                canvas_obj.setFillColor(colors.white)
                canvas_obj.setFont(font_name, 12)  # 調大字體
                canvas_obj.drawString(x_pos, 60*mm, section_title)
                
                canvas_obj.setFont(font_name, 10)  # 調大字體
                content_lines = self.wrap_text(section_content, 18, canvas_obj)
                y_pos = 50*mm
                for line in content_lines[:4]:  # 最多4行
                    if y_pos > 15*mm:
                        canvas_obj.drawString(x_pos, y_pos, line)
                        y_pos -= 3.5*mm
    
    def draw_style_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """繪製風格介紹頁 - 滿版設計"""
        # 滿版背景圖片
        if images_for_page:
            try:
                bg_img = self.create_full_bleed_image(images_for_page[0])
                canvas_obj.drawInlineImage(bg_img, 0, 0, width=210*mm, height=297*mm)
            except Exception as e:
                print(f"風格頁圖片處理錯誤: {e}")
        
        # 半透明遮罩
        canvas_obj.setFillColor(colors.black)
        canvas_obj.setFillAlpha(0.5)
        canvas_obj.rect(0, 0, 210*mm, 148*mm, fill=1, stroke=0)
        canvas_obj.setFillAlpha(1)
        
        margin = 15*mm
        
        # 標題
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 32)
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 115*mm, title)
        
        # 風格描述
        canvas_obj.setFont(font_name, 14)  # 調大字體
        description = page_data.get("style_description", "")
        if description:
            desc_lines = self.wrap_text(description, 35, canvas_obj)
            y_pos = 95*mm
            for line in desc_lines[:4]:  # 最多4行
                canvas_obj.drawString(margin, y_pos, line)
                y_pos -= 5*mm
        
        # 關鍵元素 - 雙欄
        canvas_obj.setFillColor(colors.white)
        canvas_obj.setFont(font_name, 14)
        canvas_obj.drawString(margin, 65*mm, "關鍵設計元素")
        
        canvas_obj.setFont(font_name, 12)  # 調大字體
        key_elements = page_data.get("key_elements", [])
        col1_x = margin
        col2_x = 110*mm
        y_start = 55*mm
        
        for i, element in enumerate(key_elements[:6]):  # 最多6個
            x_pos = col1_x if i % 2 == 0 else col2_x
            y_pos = y_start - (i // 2) * 6*mm
            if y_pos > 15*mm:
                canvas_obj.drawString(x_pos, y_pos, f"• {element}")
    
    def draw_summary_page(self, canvas_obj, page_data, images_for_page, template_config, font_name, layout):
        """繪製總結頁 - 滿版設計"""
        # 上半部大圖
        if images_for_page:
            try:
                img = self.crop_image_for_layout(images_for_page[0], "16:9")
                canvas_obj.drawInlineImage(img, 0, 148*mm, width=210*mm, height=149*mm)
            except Exception as e:
                print(f"總結頁圖片處理錯誤: {e}")
        
        margin = 15*mm
        
        # 標題
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 26)  # 調大字體
        title = page_data.get("title", "")
        canvas_obj.drawString(margin, 125*mm, title)
        
        # 總結內容 - 雙欄設計
        sections = [
            ("整體設計總結", page_data.get("overall_summary", "")),
            ("預算建議", page_data.get("budget_tips", "")),
            ("施工時程", page_data.get("timeline_suggestion", "")),
            ("維護保養", page_data.get("maintenance_guide", ""))
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
                    canvas_obj.setFont(font_name, 14)  # 調大字體
                    canvas_obj.drawString(x_pos, y_pos, section_title)
                    
                    canvas_obj.setFillColor(template_config["content_color"])
                    canvas_obj.setFont(font_name, 11)  # 調大字體
                    content_lines = self.wrap_text(section_content, 25, canvas_obj)
                    for j, line in enumerate(content_lines[:4]):  # 最多4行
                        canvas_obj.drawString(x_pos, y_pos - (j+1)*4*mm, line)
        
        # 聯絡資訊
        canvas_obj.setFillColor(template_config["title_color"])
        canvas_obj.setFont(font_name, 16)  # 調大字體
        contact_info = page_data.get("contact_info", "")
        if contact_info:
            canvas_obj.drawCentredString(105*mm, 25*mm, contact_info)
    
    def make_pdf_magazine(self, images, json_data, template, layout, font, output_path):
        try:
            # 調試輸出
            print(f"輸入類型檢查:")
            print(f"  images: {type(images)}")
            print(f"  json_data: {type(json_data)}")
            print(f"  template: {type(template)}")
            print(f"  font: {type(font)}")
            print(f"  output_path: {type(output_path)}")
            
            # 處理輸入參數，確保非列表參數不是列表
            if isinstance(json_data, list):
                json_data = json_data[0] if json_data else "{}"
            if isinstance(template, list):
                template = template[0] if template else "經典雜誌"
            if isinstance(layout, list):
                layout = layout[0] if layout else "版型A-經典佈局"
            if isinstance(font, list):
                font = font[0] if font else "default"
            
            # 解析JSON數據
            try:
                if isinstance(json_data, str):
                    magazine_data = json.loads(json_data)
                else:
                    magazine_data = json_data
            except json.JSONDecodeError as e:
                return (f"JSON解析錯誤: {str(e)}",)
            
            # 確保是列表格式
            if not isinstance(magazine_data, list):
                return ("錯誤: 數據格式不正確，需要陣列格式",)
            
            # 處理輸出路徑
            if isinstance(output_path, list):
                output_path = output_path[0] if output_path else "./output/magazine.pdf"
            elif not isinstance(output_path, str):
                output_path = str(output_path)
            
            # 確保輸出目錄存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            else:
                os.makedirs("./output", exist_ok=True)
                output_path = os.path.join("./output", os.path.basename(output_path))
            
            # 註冊字體
            font_name = self.register_font(font)
            
            # 獲取版型配置
            template_config = self.templates.get(template, self.templates["經典雜誌"])
            
            print(f"總共有 {len(images)} 張圖片需要處理")
            print(f"總共有 {len(magazine_data)} 個頁面需要創建")
            
            # 智能分配圖片到各頁面
            images_distribution = self.distribute_images_to_pages(magazine_data, images)
            
            # 使用Canvas創建PDF
            c = canvas.Canvas(output_path, pagesize=A4)
            
            for i, page_data in enumerate(magazine_data):
                print(f"正在處理第 {i+1} 頁: {page_data.get('page_type', 'unknown')}")
                
                # 獲取該頁面的圖片
                page_images = images_distribution.get(i, [])
                print(f"  分配到 {len(page_images)} 張圖片")
                
                # 處理圖片為PIL格式
                processed_images = []
                for img in page_images:
                    pil_img = self.tensor_to_pil(img)
                    processed_images.append(pil_img)
                
                # 繪製頁面
                self.create_custom_page(c, page_data, processed_images, template_config, font_name, layout)
                
                # 添加新頁面（除了最後一頁）
                if i < len(magazine_data) - 1:
                    c.showPage()
            
            # 保存PDF
            c.save()
            
            result = f"PDF雜誌已成功生成：{output_path}"
            print(result)
            return (result,)
            
        except Exception as e:
            error_msg = f"PDF生成錯誤: {str(e)}"
            print(error_msg)
            return (error_msg,)
    
    def distribute_images_to_pages(self, magazine_data, images):
        """智能分配圖片到各頁面，支援圖片循環重用"""
        distribution = {}
        image_index = 0
        
        if not images:
            return distribution
        
        def get_next_image():
            nonlocal image_index
            if not images:
                return None
            img = images[image_index % len(images)]  # 循環使用圖片
            image_index += 1
            return img
        
        for i, page_data in enumerate(magazine_data):
            page_type = page_data.get("page_type", "")
            
            if page_type == "cover":
                # 封面使用1張圖片
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "contents":
                # 目錄頁使用1張圖片
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "style_overview":
                # 風格介紹使用1張圖片
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
                    
            elif page_type == "room_detail":
                # 房間詳細頁使用3張圖片（確保填滿頁面）
                page_images = []
                for _ in range(3):
                    img = get_next_image()
                    if img is not None:
                        page_images.append(img)
                if page_images:
                    distribution[i] = page_images
                
            elif page_type == "summary":
                # 總結頁使用1張圖片
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
            
            else:
                # 其他頁面類型使用1張圖片
                img = get_next_image()
                if img is not None:
                    distribution[i] = [img]
        
        return distribution


NODE_CLASS_MAPPINGS = {
    "PDFMagazineGenerator": PDFMagazineGenerator,
    "PDFMagazineMaker": PDFMagazineMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDFMagazineGenerator": "PDF雜誌生成器",
    "PDFMagazineMaker": "PDF雜誌製作器"
}
