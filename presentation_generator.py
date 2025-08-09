import json
import base64
import io
import os
from PIL import Image
import google.generativeai as genai
import torch
import numpy as np

class PresentationGenerator:
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
                    "美式鄉村風", "法式古典風", "地中海風", "復古風", "新中式風", "自定風格"
                ], {"default": "現代簡約風"}),
                "custom_style": ("STRING", {"default": "", "placeholder": "當選擇自定風格時，請輸入風格描述"}),
                "custom_scene": ("STRING", {"default": "", "placeholder": "指定平面圖適用場景（如：教室、會議室、餐廳等），空白則自動判定"}),
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "json_data")
    OUTPUT_IS_LIST = (True, False)  # 只有第一個輸出是列表
    
    FUNCTION = "generate_presentation"
    CATEGORY = "presentation"
    
    def generate_presentation(self, api_key, model, design_style, custom_style, custom_scene, image):
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
            
            # 處理風格選擇
            if design_style == "自定風格" and custom_style.strip():
                effective_style = custom_style.strip()
            else:
                effective_style = design_style
            
            # 處理場景指定
            scene_instruction = ""
            if custom_scene.strip():
                scene_instruction = f"請注意，此平面圖是「{custom_scene.strip()}」的場景，"
            else:
                scene_instruction = "請先判斷此平面圖的適用場景（如住宅、辦公室、商店等），然後"
            
            # 優化後的提示詞，加入第二頁特殊格式和裝修風格
            prompt = f"""
請您作為專業室內設計師，{scene_instruction}仔細分析此平面圖並以「{effective_style}」為主要設計風格，輸出一個完整的JSON檔案，用於製作室內裝修提案簡報。

JSON結構要求：
1. 第一頁（主題頁）：
   - value: "室內裝修提案參考"
   - description: 簡短描述整體設計風格（限制15個字以內，如"現代簡約風格居家空間"）
   - image: 代表性房間的英文風格提示詞（80個token內）

2. 第二頁（提案風格概述）：
   - value: "提案風格概述"
   - description: "{effective_style}風格特色說明"
   - topic1: "動線規劃說明"
   - summary1: 針對此平面圖的動線規劃分析（35字內）
   - topic2: "設計主軸概念"
   - summary2: {effective_style}的核心設計理念（35字內）
   - topic3: "色彩與材質概念"
   - summary3: {effective_style}的色彩搭配和材質選擇（35字內）
   - topic4: "空間機能規劃說明"
   - summary4: 根據平面圖的機能空間配置說明（35字內）
   - image: {effective_style}風格的整體空間英文提示詞（80個token內）

3. 中間頁面（各房間）：
   - value: 房間類型的中文名稱（如"客廳"、"主臥室"等）
   - description: 詳細的中文裝修風格說明，包含色彩、材質、配置建議
   - image: 對應的英文ComfyUI圖片生成提示詞（80個token內）

4. 最後一頁（結尾頁）：
   - value: "謝謝聆聽本次簡報"
   - description: 風格總結與核心價值（限制15個字以內，如"打造舒適宜人居住環境"）
   - image: 整體空間感的英文提示詞（80個token內）

重要限制：
- 第一頁和最後一頁的description必須控制在15個中文字以內
- 第二頁的四個summary必須各自控制在35個中文字以內
- 請根據平面圖中的實際房間配置和大小比例進行分析
- 確保所有房間的設計風格統一為「{effective_style}」
- 英文提示詞需精準且具體，適合AI圖像生成
- 中文描述要專業且易懂，包含實用的裝修建議

請直接輸出JSON格式，不需要其他說明文字。
"""
            
            # 發送請求給 Gemini
            response = model_instance.generate_content([prompt, pil_image])
            
            # 解析回應
            response_text = response.text.strip()
            
            # 清理回應文字，移除可能的 markdown 標記
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # 解析 JSON
            presentation_data = json.loads(response_text)
            
            # 提取圖片提示詞列表
            image_prompts = []
            for item in presentation_data:
                image_prompts.append(item.get("image", ""))
            
            # 返回完整的 JSON 字串
            json_string = json.dumps(presentation_data, ensure_ascii=False, indent=2)
            
            return (image_prompts, json_string)
            
        except Exception as e:
            error_msg = f"錯誤: {str(e)}"
            return ([error_msg], error_msg)

NODE_CLASS_MAPPINGS = {
    "PresentationGenerator": PresentationGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresentationGenerator": "簡報圖案生成器"
}