import json
import os
import torch
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import io
import glob

class PresentationMaker:
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
                "json_data": ("STRING",),  # 移除multiline，確保是單純字串
                "template": (["經典簡約", "現代商務", "溫馨居家", "時尚雅緻", "清新自然"], {"default": "經典簡約"}),
                "font": (font_files, {"default": font_files[0] if font_files else "default"}),
                "output_path": ("STRING", {"default": "./output/presentation.pptx"}),
            }
        }
    
    INPUT_IS_LIST = (True, False, False, False, False)  # 只有images是列表
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    
    FUNCTION = "make_presentation"
    CATEGORY = "presentation"
    
    def __init__(self):
        # 版型配置
        self.templates = {
            "經典簡約": {
                "title_color": (44, 62, 80),
                "content_color": (52, 73, 94),
                "bg_color": (248, 249, 250),
                "accent_color": (52, 152, 219)
            },
            "現代商務": {
                "title_color": (45, 52, 54),
                "content_color": (99, 110, 114),
                "bg_color": (255, 255, 255),
                "accent_color": (0, 123, 255)
            },
            "溫馨居家": {
                "title_color": (139, 69, 19),
                "content_color": (101, 67, 33),
                "bg_color": (255, 248, 220),
                "accent_color": (205, 133, 63)
            },
            "時尚雅緻": {
                "title_color": (75, 0, 130),
                "content_color": (105, 105, 105),
                "bg_color": (245, 245, 245),
                "accent_color": (138, 43, 226)
            },
            "清新自然": {
                "title_color": (34, 139, 34),
                "content_color": (85, 107, 47),
                "bg_color": (240, 255, 240),
                "accent_color": (60, 179, 113)
            }
        }
    
    def tensor_to_pil(self, tensor):
        """將 tensor 轉換為 PIL 圖像，加入記憶體優化"""
        try:
            if isinstance(tensor, torch.Tensor):
                if tensor.dim() == 4:
                    tensor = tensor.squeeze(0)
                
                # 檢查圖片尺寸，如果太大先縮小
                height, width = tensor.shape[:2]
                max_dimension = 2048  # 限制最大尺寸
                
                if height > max_dimension or width > max_dimension:
                    scale = max_dimension / max(height, width)
                    new_height = int(height * scale)
                    new_width = int(width * scale)
                    
                    # 使用 torch 的插值來縮放，比 PIL 更節省記憶體
                    tensor = torch.nn.functional.interpolate(
                        tensor.unsqueeze(0).permute(0, 3, 1, 2),  # [1, C, H, W]
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0).permute(1, 2, 0)  # [H, W, C]
                
                # 轉換為 numpy 並限制在合理範圍
                image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
                return Image.fromarray(image_np)
            return tensor
        except Exception as e:
            print(f"Tensor轉換錯誤: {e}")
            # 創建一個預設的小圖片
            default_img = Image.new('RGB', (800, 600), color='white')
            return default_img
    
    def resize_image_to_fit(self, image, max_width, max_height, maintain_aspect=True):
        """調整圖片大小以適應指定尺寸，加入記憶體優化"""
        try:
            # 先檢查圖片尺寸，如果已經很小就不需要處理
            if image.width <= max_width and image.height <= max_height and maintain_aspect:
                return image
            
            # 計算新尺寸
            if maintain_aspect:
                # 保持比例
                ratio = min(max_width / image.width, max_height / image.height)
                new_width = int(image.width * ratio)
                new_height = int(image.height * ratio)
                
                # 如果縮放比例很小，先進行預縮放避免記憶體問題
                if ratio < 0.5:
                    # 分步縮放：先縮放到中間尺寸
                    intermediate_ratio = 0.7
                    intermediate_width = int(image.width * intermediate_ratio)
                    intermediate_height = int(image.height * intermediate_ratio)
                    image = image.resize((intermediate_width, intermediate_height), Image.Resampling.LANCZOS)
                    
                    # 重新計算最終比例
                    ratio = min(max_width / image.width, max_height / image.height)
                    new_width = int(image.width * ratio)
                    new_height = int(image.height * ratio)
                
            else:
                # 16:9 滿版裁切
                target_ratio = 16/9
                img_ratio = image.width / image.height
                
                # 先檢查是否需要大幅縮放
                scale_factor = min(max_width / image.width, max_height / image.height)
                if scale_factor < 0.5:
                    # 先縮放到接近目標尺寸
                    temp_width = int(max_width * 1.2)
                    temp_height = int(max_height * 1.2)
                    image = image.resize((temp_width, temp_height), Image.Resampling.LANCZOS)
                
                if img_ratio > target_ratio:
                    # 圖片太寬，裁切寬度
                    new_height = image.height
                    new_width = int(new_height * target_ratio)
                    x = (image.width - new_width) // 2
                    image = image.crop((x, 0, x + new_width, new_height))
                else:
                    # 圖片太高，裁切高度
                    new_width = image.width
                    new_height = int(new_width / target_ratio)
                    y = (image.height - new_height) // 2
                    image = image.crop((0, y, new_width, y + new_height))
                
                new_width = max_width
                new_height = max_height
            
            # 最終調整大小
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
        except MemoryError:
            # 記憶體不足時的降級處理
            print(f"記憶體不足，嘗試降級處理圖片 {image.width}x{image.height}")
            
            # 使用更激進的縮放策略
            if maintain_aspect:
                ratio = min(max_width / image.width, max_height / image.height)
                # 進一步減小比例以節省記憶體
                safe_ratio = min(ratio * 0.8, 0.5)
                new_width = int(image.width * safe_ratio)
                new_height = int(image.height * safe_ratio)
            else:
                # 直接設置為目標尺寸的較小版本
                new_width = int(max_width * 0.8)
                new_height = int(max_height * 0.8)
            
            # 使用較低品質的縮放算法以節省記憶體
            return image.resize((new_width, new_height), Image.Resampling.NEAREST)
            
        except Exception as e:
            print(f"圖片處理錯誤: {e}")
            # 返回原始圖片的小版本
            return image.resize((min(800, image.width), min(600, image.height)), Image.Resampling.NEAREST)
    
    def add_image_to_slide(self, slide, image, left, top, width, height):
        """將圖片添加到投影片"""
        img_stream = io.BytesIO()
        image.save(img_stream, format='PNG')
        img_stream.seek(0)
        
        pic = slide.shapes.add_picture(img_stream, left, top, width, height)
        return pic
    
    def generate_unique_filename(self, output_path):
        """生成唯一檔名，避免覆蓋現有檔案"""
        if not os.path.exists(output_path):
            return output_path
        
        # 分離檔案路徑、名稱和副檔名
        dir_path = os.path.dirname(output_path)
        filename = os.path.basename(output_path)
        name, ext = os.path.splitext(filename)
        
        # 尋找可用的檔名
        counter = 1
        while True:
            new_filename = f"{name}{counter:02d}{ext}"
            new_path = os.path.join(dir_path, new_filename)
            if not os.path.exists(new_path):
                return new_path
            counter += 1
            
            # 防止無限循環，最多嘗試100次
            if counter > 100:
                import time
                timestamp = int(time.time())
                new_filename = f"{name}_{timestamp}{ext}"
                return os.path.join(dir_path, new_filename)
    
    def make_presentation(self, images, json_data, template, font, output_path):
        try:
            # 根據ComfyUI規範，當INPUT_IS_LIST=True時，所有參數都是列表
            # 從列表中提取實際值
            json_string = json_data[0] if isinstance(json_data, list) and json_data else "{}"
            template_name = template[0] if isinstance(template, list) and template else "經典簡約"
            font_name = font[0] if isinstance(font, list) and font else "default"
            output_file = output_path[0] if isinstance(output_path, list) and output_path else "./presentation.pptx"
            
            # 處理圖片列表 - 展開所有批次圖片
            all_images = []
            for img_batch in images:  # images 是圖片批次的列表
                if isinstance(img_batch, torch.Tensor):
                    # 如果是批次張量，分離每張圖片
                    if img_batch.dim() == 4:  # [batch, height, width, channels]
                        for i in range(img_batch.shape[0]):
                            all_images.append(img_batch[i])
                    else:  # 單張圖片
                        all_images.append(img_batch)
                else:
                    all_images.append(img_batch)
            
            # 解析 JSON 數據
            try:
                presentation_data = json.loads(json_string)
            except json.JSONDecodeError as e:
                return (f"JSON 解析錯誤: {str(e)}，接收到的數據: {str(json_string)[:200]}",)
            
            # 檢查圖片數量是否與 JSON 數據匹配
            if len(all_images) != len(presentation_data):
                return (f"圖片數量 ({len(all_images)}) 與 JSON 數據項目 ({len(presentation_data)}) 不匹配",)
            
            # 創建新的簡報
            prs = Presentation()
            
            # 設置簡報尺寸為 16:9
            prs.slide_width = Inches(13.33)
            prs.slide_height = Inches(7.5)
            
            # 獲取版型配置
            template_config = self.templates.get(template_name, self.templates["經典簡約"])
            
            # 處理每一頁
            for i, (image, data_item) in enumerate(zip(all_images, presentation_data)):
                print(f"正在處理第 {i+1} 頁...")
                
                # 提取數據
                value = data_item.get("value", "")
                description = data_item.get("description", "")
                
                # 跳過空值
                if not value:
                    continue
                
                # 添加新投影片
                slide_layout = prs.slide_layouts[6]  # 空白版面
                slide = prs.slides.add_slide(slide_layout)
                
                # 設置背景色
                background = slide.background
                fill = background.fill
                fill.solid()
                fill.fore_color.rgb = RGBColor(*template_config["bg_color"])
                
                # 轉換圖片
                print(f"轉換圖片...")
                pil_image = self.tensor_to_pil(image)
                print(f"圖片尺寸: {pil_image.width}x{pil_image.height}")
                
                # 第一頁和最後一頁 - 滿版設計
                if i == 0 or i == len(all_images) - 1:
                    print(f"處理需要縮放到滿版...")
                    # 圖片滿版 16:9 - 使用較小的尺寸以節省記憶體
                    slide_width_px = 1920  # 降低解析度
                    slide_height_px = 1080
                    
                    processed_image = self.resize_image_to_fit(
                        pil_image, 
                        slide_width_px, 
                        slide_height_px, 
                        maintain_aspect=False
                    )
                    print(f"縮放後尺寸: {processed_image.width}x{processed_image.height}")
                    
                    # 添加背景圖片
                    self.add_image_to_slide(
                        slide, processed_image, 
                        0, 0, 
                        prs.slide_width, prs.slide_height
                    )
                    
                    # 使用圖層堆疊方式創建半透明背景效果
                    # 文字區域的位置與大小
                    text_left = Inches(1)
                    text_top = Inches(2.5)
                    text_width = Inches(11.33)
                    text_height = Inches(2.5)
                    
                    # ======= 底層：半透明背景框（用深色模擬） =======
                    from pptx.enum.shapes import MSO_SHAPE
                    bg_box = slide.shapes.add_shape(
                        MSO_SHAPE.RECTANGLE,
                        text_left,
                        text_top,
                        text_width,
                        text_height
                    )
                    bg_box.fill.solid()
                    bg_box.fill.fore_color.rgb = RGBColor(40, 40, 40)  # 模擬70%黑色透明
                    bg_box.line.fill.background()  # 無邊框線
                    
                    # ======= 上層：文字框，背景透明 =======
                    text_box = slide.shapes.add_textbox(
                        text_left, text_top, text_width, text_height
                    )
                    text_frame = text_box.text_frame
                    text_frame.clear()
                    text_frame.margin_left = Inches(0.2)
                    text_frame.margin_right = Inches(0.2)
                    text_frame.margin_top = Inches(0.2)
                    text_frame.margin_bottom = Inches(0.2)
                    
                    # 標題文字
                    title_p = text_frame.add_paragraph()
                    title_p.text = str(value)
                    title_p.alignment = PP_ALIGN.CENTER
                    title_p.font.size = Pt(44)
                    title_p.font.color.rgb = RGBColor(255, 255, 255)
                    title_p.font.bold = True
                    
                    # 副標題
                    if description:
                        subtitle_p = text_frame.add_paragraph()
                        subtitle_p.text = str(description)
                        subtitle_p.alignment = PP_ALIGN.CENTER
                        subtitle_p.font.size = Pt(24)
                        subtitle_p.font.color.rgb = RGBColor(240, 240, 240)
                    
                    # 設定文字框背景透明
                    text_box.fill.background()  # 透明背景
                    text_box.line.fill.background()  # 無邊框
                
                else:
                    print(f"處理內容頁 {i+1}，保持原圖尺寸...")
                    # 內容頁 - 左右排列，保持原圖尺寸
                    # 不進行任何縮放，直接使用原圖
                    processed_image = pil_image
                    
                    # 計算圖片在簡報中的實際顯示尺寸
                    display_width = Inches(5.5)
                    display_height = Inches(4.6)
                    
                    # 如果原圖太大，只在這裡進行輕微調整以適應顯示區域
                    if pil_image.width > 1200 or pil_image.height > 1000:
                        print(f"圖片過大 ({pil_image.width}x{pil_image.height})，進行適度縮放...")
                        # 只有在圖片明顯過大時才縮放
                        processed_image = self.resize_image_to_fit(
                            pil_image, 1200, 1000, maintain_aspect=True
                        )
                        print(f"縮放後尺寸: {processed_image.width}x{processed_image.height}")
                    else:
                        print(f"圖片尺寸適中，保持原尺寸: {pil_image.width}x{pil_image.height}")
                    
                    # 添加圖片 (右側)
                    print(f"添加圖片到簡報...")
                    self.add_image_to_slide(
                        slide, processed_image,
                        Inches(7.3), Inches(1.5),
                        display_width, display_height
                    )
                    
                    # 添加標題 (左上)
                    title_box = slide.shapes.add_textbox(
                        Inches(0.5), Inches(0.5),
                        Inches(6.5), Inches(1)
                    )
                    title_frame = title_box.text_frame
                    title_p = title_frame.add_paragraph()
                    title_p.text = str(value)
                    title_p.font.size = Pt(32)
                    title_p.font.color.rgb = RGBColor(*template_config["title_color"])
                    title_p.font.bold = True
                    
                    # 添加內容 (左下)
                    content_box = slide.shapes.add_textbox(
                        Inches(0.5), Inches(2),
                        Inches(6.5), Inches(4.5)
                    )
                    content_frame = content_box.text_frame
                    content_frame.word_wrap = True
                    
                    content_p = content_frame.add_paragraph()
                    content_p.text = str(description) if description else ""
                    content_p.font.size = Pt(16)
                    content_p.font.color.rgb = RGBColor(*template_config["content_color"])
                    content_p.line_spacing = 1.5
            
            # 確保輸出目錄存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 生成唯一檔名以避免覆蓋
            unique_output_path = self.generate_unique_filename(output_file)
            
            # 保存簡報
            prs.save(unique_output_path)
            
            return (f"簡報已成功生成：{unique_output_path}",)
            
        except Exception as e:
            error_msg = f"生成簡報時發生錯誤: {str(e)}"
            print(f"詳細錯誤: {e}")  # 用於調試
            import traceback
            print(f"錯誤追蹤: {traceback.format_exc()}")
            return (error_msg,)

NODE_CLASS_MAPPINGS = {
    "PresentationMaker": PresentationMaker
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresentationMaker": "簡報製作器"
}