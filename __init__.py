from .presentation_generator import NODE_CLASS_MAPPINGS as generator_mappings, NODE_DISPLAY_NAME_MAPPINGS as generator_display_mappings
from .presentation_maker import NODE_CLASS_MAPPINGS as maker_mappings, NODE_DISPLAY_NAME_MAPPINGS as maker_display_mappings
from .pdf_magazine_generator import NODE_CLASS_MAPPINGS as pdf_mappings, NODE_DISPLAY_NAME_MAPPINGS as pdf_display_mappings
from .llm_manager import NODE_CLASS_MAPPINGS as llm_mappings, NODE_DISPLAY_NAME_MAPPINGS as llm_display_mappings

# Merge node mappings
NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(generator_mappings)
NODE_CLASS_MAPPINGS.update(maker_mappings)
NODE_CLASS_MAPPINGS.update(pdf_mappings)
NODE_CLASS_MAPPINGS.update(llm_mappings)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(generator_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(maker_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(pdf_display_mappings)
NODE_DISPLAY_NAME_MAPPINGS.update(llm_display_mappings)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
