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
                "image": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "json_data")
    OUTPUT_IS_LIST = (True, False)  # 只有第一個輸出是列表
    
    FUNCTION = "generate_presentation"
    CATEGORY = "presentation"
    
    def generate_presentation(self, api_key, model, image):
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
            
            # 優化後的提示詞，限制副標題長度
            prompt = """
請您作為專業室內設計師，仔細分析此平面圖並輸出一個完整的JSON檔案，用於製作室內裝修提案簡報。

JSON結構要求：
1. 第一頁（主題頁）：
   - value: "室內裝修提案參考"
   - description: 簡短描述整體設計風格（限制15個字以內，如"現代簡約風格居家空間"）
   - image: 代表性房間的英文風格提示詞（80個token內）

2. 中間頁面（各房間）：
   - value: 房間類型的中文名稱（如"客廳"、"主臥室"等）
   - description: 詳細的中文裝修風格說明，包含色彩、材質、配置建議
   - image: 對應的英文ComfyUI圖片生成提示詞（80個token內）

3. 最後一頁（結尾頁）：
   - value: "謝謝聆聽本次簡報"
   - description: 風格總結與核心價值（限制15個字以內，如"打造舒適宜人居住環境"）
   - image: 整體空間感的英文提示詞（80個token內）

重要限制：
- 第一頁和最後一頁的description必須控制在15個中文字以內，避免簡報跑版
- 請根據平面圖中的實際房間配置和大小比例進行分析
- 確保所有房間的設計風格統一協調
- 英文提示詞需精準且具體，適合AI圖像生成
- 中文描述要專業且易懂，包含實用的裝修建議
- 不需要描述具體尺寸數字，重點在於空間配置和風格營造

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