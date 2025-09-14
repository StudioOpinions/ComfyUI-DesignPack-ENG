# ComfyUI-DesignPack

A ComfyUI custom node package for generating professional interior design presentations using AI. This package integrates with Google's Gemini API to analyze floor plans and automatically generate PowerPoint presentations with AI-generated content and images.

## Features

- Automatic interior design presentation generation from floor plans
- AI content analysis using Google Gemini API to produce structured presentation data
- Multiple presentation templates in five different styles
- Automated integration of generated images into slides
- PDF magazine creation
- Unified LLM API management

## Nodes

1. **PresentationGenerator** – analyzes floor plans and produces presentation data and image prompts
2. **PresentationMaker** – creates PowerPoint presentations from data and images
3. **PDFMagazineGenerator** – generates PDF design magazines
4. **LLMManager** – manages configuration for multiple LLM APIs

## Installation

1. Copy this folder into the `custom_nodes` directory of ComfyUI.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI.

## Usage

### Basic workflow

1. **Configure the LLM API**
   - Enter your Google Gemini API key in the LLMManager node.
   - The key is saved to `config.json`.

2. **Prepare a floor plan**
   - Load a floor plan image into ComfyUI.
   - Connect it to the `image` input of the PresentationGenerator node.

3. **Generate presentation data**
   - PresentationGenerator outputs:
     - `image_prompts`: prompts for image generation
     - `json_data`: structured presentation data

4. **Connect prompt nodes**
   - Connect `image_prompts` to a CLIPTextEncode node to generate design images.

5. **Create the presentation**
   - Connect generated images and `json_data` to the PresentationMaker node.
   - Set an output path and choose a template.
   - Execute the node to produce a PowerPoint file.

### Connection overview

- `image_prompts` → CLIPTextEncode: feed prompts into the text encoder
- `json_data` → PresentationMaker: supply structured data to the maker
- Generated images → PresentationMaker: provide images for slides

### Configuration options

- **Output path**: where the PowerPoint file is saved
- **Template**: choose the visual style
  - Classic Minimalist
  - Modern Business
  - Warm Home
  - Fashionable Elegant
  - Fresh Natural

## Example workflow

```
Floor Plan Image → PresentationGenerator → image_prompts → CLIPTextEncode → Image Generation
                                      ↓
                                  json_data → PresentationMaker ← Generated Images
                                                     ↓
                                              PowerPoint Output
```

## File structure

```
ComfyUI-DesignPack/
├── __init__.py
├── presentation_generator.py
├── presentation_maker.py
├── llm_manager.py
├── requirements.txt
├── config.json (auto-generated)
└── fonts/
    └── cht.ttf
```

## Notes

- Requires a valid Google Gemini API key
- Images should be PNG, JPG, or other common formats
- High-resolution floor plans yield better analysis results
- Generated presentations require fonts that support all included characters

## Troubleshooting

- **Memory errors**: reduce the resolution of input images
- **API errors**: verify your API key and network connection
- **Presentation creation fails**: ensure the output path is writable and has enough space
- **Font issues**: check that required fonts exist in the `fonts` directory
