import json
import os
import requests
import google.generativeai as genai

class LLMManager:
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
    
    def test_ollama(self, model):
        """Test Ollama connection"""
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": model,
                "prompt": "hello",
                "stream": False
            }
            response = requests.post(url, json=data, timeout=10)
            if response.status_code == 200:
                return True, "Ollama available"
            else:
                return False, f"Ollama connection failed: HTTP {response.status_code}"
        except Exception as e:
            return False, f"Ollama test failed: {str(e)}"
    
    def test_gemini(self, api_key, model):
        """Test Gemini connection"""
        try:
            genai.configure(api_key=api_key)
            model_instance = genai.GenerativeModel(model)
            response = model_instance.generate_content("hello")
            if response.text:
                return True, "Gemini available"
            else:
                return False, "Gemini response was empty"
        except Exception as e:
            return False, f"Gemini test failed: {str(e)}"
    
    def test_openai(self, api_key, model, url):
        """Test OpenAI connection"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 50
            }
            
            response = requests.post(f"{url}/chat/completions", 
                                   headers=headers, json=data, timeout=10)
            if response.status_code == 200:
                return True, "OpenAI available"
            else:
                return False, f"OpenAI connection failed: HTTP {response.status_code}"
        except Exception as e:
            return False, f"OpenAI test failed: {str(e)}"

    @classmethod
    def INPUT_TYPES(cls):
        # Load saved configuration
        instance = cls()
        config = instance.load_config()
        
        return {
            "required": {
                "ollama_model": ("STRING", {
                    "default": config.get("ollama_model", "gemma3:12b"),
                    "placeholder": "Ollama model name"
                }),
                "gemini_model": ("STRING", {
                    "default": config.get("gemini_model", "gemini-2.5-flash"),
                    "placeholder": "Gemini model name"
                }),
                "gemini_api": ("STRING", {
                    "default": config.get("gemini_api", ""),
                    "placeholder": "Gemini API Key"
                }),
                "openai_model": ("STRING", {
                    "default": config.get("openai_model", "gpt-4o-mini"),
                    "placeholder": "OpenAI model name"
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
            # Save configuration
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
            
            # Test Ollama
            if ollama_model:
                success, message = self.test_ollama(ollama_model)
                results.append(f"Ollama ({ollama_model}): {message}")
            
            # Test Gemini
            if gemini_api and gemini_model:
                success, message = self.test_gemini(gemini_api, gemini_model)
                results.append(f"Gemini ({gemini_model}): {message}")
            
            # Test OpenAI
            if openai_key and openai_model and openai_url:
                success, message = self.test_openai(openai_key, openai_model, openai_url)
                results.append(f"OpenAI ({openai_model}): {message}")
            
            return ("\n".join(results),)
            
        except Exception as e:
            return (f"Error occurred during testing: {str(e)}",)

NODE_CLASS_MAPPINGS = {
    "LLMManager": LLMManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMManager": "Global LLM Manager"
}
