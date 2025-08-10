import json
import os
import requests
import google.generativeai as genai

class LLMManager:
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
    
    def test_ollama(self, model):
        """測試Ollama連接"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": "你好",
                "stream": False
            }
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return True, "Ollama可用"
            else:
                return False, f"Ollama連接失敗: HTTP {response.status_code}"
        except Exception as e:
            return False, f"Ollama測試失敗: {str(e)}"
    
    def test_gemini(self, api_key, model):
        """測試Gemini連接"""
        try:
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content("你好")
            if response.text:
                return True, "Gemini可用"
            else:
                return False, "Gemini回應為空"
        except Exception as e:
            return False, f"Gemini測試失敗: {str(e)}"
    
    def test_openai(self, api_key, model, url):
        """測試OpenAI連接"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "你好"}],
                "max_tokens": 50
            }
            
            response = requests.post(f"{url}/chat/completions", 
                                   headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return True, "OpenAI可用"
            else:
                return False, f"OpenAI連接失敗: HTTP {response.status_code}"
        except Exception as e:
            return False, f"OpenAI測試失敗: {str(e)}"

    @classmethod
    def INPUT_TYPES(cls):
        # 載入已儲存的配置
        instance = cls()
        config = instance.load_config()
        
        return {
            "required": {
                "ollama_model": ("STRING", {
                    "default": config.get("ollama_model", "gemma3:12b"),
                    "placeholder": "Ollama模型名稱"
                }),
                "gemini_model": ("STRING", {
                    "default": config.get("gemini_model", "gemini-2.5-flash"),
                    "placeholder": "Gemini模型名稱"
                }),
                "gemini_api": ("STRING", {
                    "default": config.get("gemini_api", ""),
                    "placeholder": "Gemini API Key"
                }),
                "openai_model": ("STRING", {
                    "default": config.get("openai_model", "gpt-4o-mini"),
                    "placeholder": "OpenAI模型名稱"
                }),
                "openai_url": ("STRING", {
                    "default": config.get("openai_url", "https://api.openai.com/v1"),
                    "placeholder": "OpenAI API URL"
                }),
                "openai_key": ("STRING", {
                    "default": config.get("openai_key", ""),
                    "placeholder": "OpenAI API Key"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("test_results",)
    
    FUNCTION = "test_llm_connections"
    CATEGORY = "llm_manager"
    
    def test_llm_connections(self, ollama_model, gemini_model, gemini_api, 
                           openai_model, openai_url, openai_key):
        try:
            # 保存配置
            config = {
                "ollama_model": ollama_model,
                "gemini_model": gemini_model,
                "gemini_api": gemini_api,
                "openai_model": openai_model,
                "openai_url": openai_url,
                "openai_key": openai_key
            }
            self.save_config(config)
            
            results = []
            
            # 測試Ollama
            if ollama_model:
                success, message = self.test_ollama(ollama_model)
                results.append(f"Ollama ({ollama_model}): {message}")
            
            # 測試Gemini
            if gemini_api and gemini_model:
                success, message = self.test_gemini(gemini_api, gemini_model)
                results.append(f"Gemini ({gemini_model}): {message}")
            
            # 測試OpenAI
            if openai_key and openai_model and openai_url:
                success, message = self.test_openai(openai_key, openai_model, openai_url)
                results.append(f"OpenAI ({openai_model}): {message}")
            
            return ("\n".join(results),)
            
        except Exception as e:
            return (f"測試過程發生錯誤: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "LLMManager": LLMManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMManager": "全局LLM管理器"
}