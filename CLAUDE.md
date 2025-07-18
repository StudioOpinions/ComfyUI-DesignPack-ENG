# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ComfyUI custom node package called "ComfyUI-DesignPack" that provides presentation generation capabilities for interior design projects. The package integrates with Google's Gemini API to analyze floor plans and generate PowerPoint presentations with AI-generated content and images.

## Architecture

The codebase follows ComfyUI's custom node architecture with two main components:

### Core Components

1. **PresentationGenerator** (`presentation_generator.py`):
   - Analyzes floor plan images using Google Gemini API
   - Generates structured JSON data for presentation content
   - Handles API key management and persistence
   - Returns image prompts and structured presentation data

2. **PresentationMaker** (`presentation_maker.py`):
   - Creates PowerPoint presentations using python-pptx
   - Processes AI-generated images and integrates them into slides
   - Provides multiple presentation templates and styling options
   - Handles memory optimization for large images

3. **Node Registration** (`__init__.py`):
   - Registers both nodes with ComfyUI
   - Merges node class mappings and display name mappings
   - Follows ComfyUI's standard node registration pattern

### Key Dependencies

- **google-generativeai**: For Gemini API integration
- **python-pptx**: For PowerPoint generation
- **pillow**: For image processing
- **numpy**: For numerical operations
- **torch**: For ComfyUI tensor operations

## Configuration

### API Key Management
- Gemini API keys are stored in `config.json`
- Keys are automatically saved when entered in the PresentationGenerator node
- The system gracefully handles missing API keys

### Font Support
- Custom fonts are stored in the `fonts/` directory
- Currently includes `cht.ttf` for Chinese text support
- Font selection is available in the PresentationMaker node

## Data Flow

1. **Input**: Floor plan image → PresentationGenerator
2. **Processing**: Gemini API analyzes image and generates structured JSON
3. **Output**: JSON data + image prompts → PresentationMaker
4. **Final**: AI-generated images + JSON → PowerPoint presentation

## Memory Management

The codebase includes extensive memory optimization:
- Image resizing with progressive scaling
- Tensor-to-PIL conversion with size limits
- Memory error handling with fallback strategies
- Unique filename generation to prevent overwrites

## Template System

Five presentation templates are available:
- 經典簡約 (Classic Minimalist)
- 現代商務 (Modern Business)
- 溫馨居家 (Warm Home)
- 時尚雅緻 (Fashionable Elegant)
- 清新自然 (Fresh Natural)

Each template defines color schemes for titles, content, backgrounds, and accents.

## Development Guidelines

### Working with Images
- Always use the `tensor_to_pil()` method for tensor conversion
- Consider memory limits when processing large images
- Use `resize_image_to_fit()` for consistent image sizing

### Node Development
- Follow ComfyUI's INPUT_TYPES format for parameter definitions
- Use INPUT_IS_LIST appropriately for batch processing
- Implement proper error handling with user-friendly messages

### API Integration
- Store API keys securely in config.json
- Implement proper error handling for network requests
- Use structured prompts for consistent AI responses

## Testing

This project doesn't include formal test files. Testing is done through ComfyUI's node execution system. When making changes:

1. Test both nodes independently
2. Verify the complete workflow from floor plan to presentation
3. Test with various image sizes and formats
4. Validate different presentation templates

## Common Issues

- **Memory errors**: Reduce image sizes or use progressive scaling
- **API failures**: Check API key validity and network connectivity
- **JSON parsing errors**: Ensure Gemini responses are properly formatted
- **Font issues**: Verify font files exist in the fonts directory

## File Structure

```
ComfyUI-DesignPack/
├── __init__.py              # Node registration
├── presentation_generator.py # Gemini API integration
├── presentation_maker.py     # PowerPoint generation
├── config.json              # API key storage
├── requirements.txt         # Dependencies
└── fonts/                   # Font files
    └── cht.ttf
```