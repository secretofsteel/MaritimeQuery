"""Constants copied from the original Streamlit implementation."""

from __future__ import annotations
import os
import json
from pathlib import Path

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200


# Python slicing extraction configuration
# Fuzzy matching threshold for finding section boundaries in raw text (0-1 scale)
# Lower values = more lenient matching, higher values = stricter matching
# Recommended range: 0.80-0.90 for maritime documents with formatting variations
PYTHON_SLICING_FUZZY_THRESHOLD = 0.80

# Debug mode toggle (can be controlled via environment variable or admin UI)
DEBUG_RAG = os.getenv("DEBUG_RAG", "false").lower() == "true"

GEMINI_SCHEMA = {
    "type": "object",
    "properties": {
        "filename": {"type": "string"},
        "doc_type": {"type": "string"},
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["name", "content"],
            },
        },
        "category": {"type": "string"},
        "form_number": {"type": "string"},
        "normalized_topic": {"type": "string"},
        "hierarchy": {"type": "array", "items": {"type": "string"}},
        "references": {
            "type": "object",
            "properties": {
                "forms": {"type": "array", "items": {"type": "string"}},
                "procedures": {"type": "array", "items": {"type": "string"}},
                "regulations": {"type": "array", "items": {"type": "string"}},
                "policies": {"type": "array", "items": {"type": "string"}},
                "reports": {"type": "array", "items": {"type": "string"}},
                "chapters": {"type": "array", "items": {"type": "string"}},
                "sections": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": ["filename", "doc_type", "title"],
}

# ==============================================================================
# FORM CATEGORIES - Now loaded from JSON with fallback
# ==============================================================================

def load_form_categories() -> dict[str, str]:
    """
    Load form categories from JSON file, with fallback to hardcoded constants.
    
    Returns:
        Dictionary mapping form codes to category descriptions
    """
    # Hardcoded fallback (original FORM_CATEGORIES)
    FALLBACK_CATEGORIES = {
        "Z": "Miscellaneous (Chapter 3)",
        "C": "Crew",
        "CBO": "Cargo/Ballast Operations",
        "M": "Maintenance",
        "N": "Navigation",
        "HR": "Human Resources",
        "E": "Engine Room Operations",
        "EN": "Environmental",
        "DA": "Drug & Alcohol",
        "P": "Safe Working Procedures",
        "S": "Health & Hygiene",
        "DR": "Drills",
        "NCR": "Non-Compliance Reporting",
        "D": "Document Control",
        "A": "Audits",
        "MOC": "Management of Change",
        "RA": "Risk Assessment",
        "Q": "Quality Control",
        "CS": "Cyber Security",
    }
    
    # Try to load from JSON
    config_path = Path(__file__).parent.parent / "config" / "form_categories.json"
    
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                categories = json.load(f)
            
            # Validate it's a dict with string keys/values
            if isinstance(categories, dict) and all(
                isinstance(k, str) and isinstance(v, str) 
                for k, v in categories.items()
            ):
                return categories
            else:
                print(f"WARNING: Invalid format in {config_path}, using fallback")
                return FALLBACK_CATEGORIES
        
        except (json.JSONDecodeError, OSError) as exc:
            print(f"WARNING: Failed to load {config_path}: {exc}, using fallback")
            return FALLBACK_CATEGORIES
    
    # JSON doesn't exist yet, return fallback
    return FALLBACK_CATEGORIES


def save_form_categories(categories: dict[str, str]) -> bool:
    """
    Save form categories to JSON file.
    
    Args:
        categories: Dictionary mapping form codes to descriptions
    
    Returns:
        True if saved successfully, False otherwise
    """
    config_path = Path(__file__).parent.parent / "config" / "form_categories.json"
    
    try:
        # Create config directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSON with nice formatting
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(categories, f, indent=2, ensure_ascii=False)
        
        return True
    
    except (OSError, TypeError) as exc:
        print(f"ERROR: Failed to save form categories to {config_path}: {exc}")
        return False


def get_form_categories_path() -> Path:
    """Get the path to the form categories JSON file."""
    return Path(__file__).parent.parent / "config" / "form_categories.json"


# Load form categories (from JSON or fallback)
FORM_CATEGORIES = load_form_categories()

# ==============================================================================
# REST OF CONSTANTS
# ==============================================================================

EXTRACT_SYSTEM_PROMPT = (
    "You are a maritime document analysis model. Your task is to extract structured information in STRICT JSON format based on the provided schema. Be sure to use double quotes for the keys and values of the JSON file"
    "Return ONLY the JSON object. Do not include any surrounding text, markdown code blocks (like ```json), or explanations."
    "Ensure all string values within the JSON are properly escaped according to JSON standards (e.g., `\\'` and '\"' for quotes, `\\\\` for backslashes, `\\n` for new lines, `\\t` for tabs)."
    "Always ensure to properly close the JSON output with the required brackets. Never forget to add the ',' delimiter where it is needed"
    "Detect and preserve tables and lists within section content. Represent them as accurately as possible in the raw text content."
    "If nested hierarchy of sections exists, include it as an array of strings in 'hierarchy', (for example 'Chapter 4', '1.1 Bunkering', '1.1.1 Measuring the bunkers', etc.)"
    "Map shorthand codes like 'Form C 002' or 'Checklist EN 002' to full category names using the provided map."
    "Identify distinct sections within the document. For EACH identified section, provide its 'name' and the full 'content' belonging to that section. The 'content' should include all relevant text, lists, tables, and details associated with the section name."
    "Be precise in separating content belonging to different sections."
    "If the document is a Form or a Checklist ensure to start the title with the Form or Checklist code and number then follow with its title, (eg. 'C 004 - Briefing of Masters and Senior Offiers' or 'CBO 015 - Cargo Operation Checklist')"
    "Identify and list any explicit cross-references to other documents, forms, procedures, policies, regulations, reports, their chapters and/or sections mentioned in the text."
    "Use the 'references' object in the schema, listing full names/numbers (e.g., 'Form C 002b', 'Ballast Water Management Procedure', 'SOLAS Chapter V'). Be specific and include the complete name/number as it appears or is logically inferred."
    "IMPORTANT: Any form, procedure,regulation, etc. that goes into the references should be included only ONCE!! DO NOT INSERT REPEATED ENTRIES!!!!"
    "Adhere strictly to the provided JSON schema, including all required properties and data types."
    "Do not include trailing commas after the last item in arrays or objects."
    "CRITICAL CONTENT RULES:"
    "- Extract content as plain text only, NO formatting preservation"
    "- Replace all sequences of dots (....) with single ellipsis (...)"
    "- Replace all tabs with single space"
    "- Replace all multiple spaces with single space"
    "- Do NOT try to preserve visual alignment or formatting"
    "- Content should be clean, readable text suitable for search"
    "Title extraction rules:"
    "- The filename should be included as-is in the 'filename' field"
    "- If it is a Form or Checklist, the 'title' must start with the form/checklist code and number, followed by a hyphen and the title. Use the filename first to help infer the code/number and title"
    "- For procedures, policies, regulations, and manuals, the 'title' should be the main title of the document as prominently displayed, if not, use the filename to help infer the title"
    "- Generally, the document title should be taken from the main heading/header or prominent title text or filename if no title is found, or combination of both if needed for clarity"
    "- Rule of Thumb: in case of forms/checklists filename trumps extracted title, in case of procedures/policies/regulations/manuals extracted title trumps filename"
    "- If using filename as title, remove file extension and replace underscores/hyphens with spaces for readability"
    "doc_type options:"
    "define the doc type strictly from this list:"
    "- Form"
    "- Checklist"
    "- Procedure"
    "- Regulation"
    "- Policy"
    "- Manual"
    "Use 'Manual' only when it is a set of instructions or equipment manual"
    "For any document that is part of company's Management System, then label it 'Procedure'"
)


CONFIDENCE_HIGH_THRESHOLD = 75
CONFIDENCE_MEDIUM_THRESHOLD = 55

# Context-aware conversation settings
MAX_CONTEXT_TURNS = 6  # Hard reset after this many exchanges
CONTEXT_HISTORY_WINDOW = 5  # Number of recent Q&A pairs to include in prompt

# Hierarchical Retrieval Configuration
HIERARCHICAL_MAX_SECTIONS = 2  # Maximum sections to retrieve per query
HIERARCHICAL_MAX_DEPTH = 3  # Maximum recursion depth in section tree
HIERARCHICAL_MIN_CONTEXT_TOKENS = 500  # Fallback to chunks if less than this
HIERARCHICAL_MAX_CONTEXT_TOKENS = 8000  # Token budget per query

__all__ = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "DEBUG_RAG",
    "GEMINI_SCHEMA",
    "FORM_CATEGORIES",
    "EXTRACT_SYSTEM_PROMPT",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "MAX_CONTEXT_TURNS",
    "CONTEXT_HISTORY_WINDOW",
    "HIERARCHICAL_MAX_SECTIONS",
    "HIERARCHICAL_MAX_DEPTH",
    "HIERARCHICAL_MIN_CONTEXT_TOKENS",
    "HIERARCHICAL_MAX_CONTEXT_TOKENS",
    "load_form_categories",
    "save_form_categories",
    "get_form_categories_path",
]

