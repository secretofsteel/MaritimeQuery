"""Constants copied from the original Streamlit implementation."""

from __future__ import annotations

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200

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

FORM_CATEGORIES = {
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

EXTRACT_SYSTEM_PROMPT = (
    "You are a maritime document analysis model. Your task is to extract structured information in STRICT JSON format based on the provided schema. Be sure to use double quotes for the keys and values of the JSON file"
    "Return ONLY the JSON object. Do not include any surrounding text, markdown code blocks (like ```json), or explanations."
    "Ensure all string values within the JSON are properly escaped according to JSON standards (e.g., `\\'` and '\"' for quotes, `\\\\` for backslashes, `\\n` for new lines, `\\t` for tabs)."
    "Always ensure to properly close the JSON output with the required brackets. Never forget to add the ',' delimiter where it is needed"
    "Detect and preserve tables and lists within section content. Represent them as accurately as possible in the raw text content."
    "If hierarchy of sections exists, include it as an array of strings in 'hierarchy', (for example 'Chapter 4', '1.1 Bunkering', '1.1.1 Measuring the bunkers', etc.)"
    "Map shorthand codes like 'Form C 002' or 'Checklist EN 002' to full category names using the provided map."
    "Identify distinct sections within the document. For EACH identified section, provide its 'name' and the full 'content' belonging to that section. The 'content' should include all relevant text, lists, tables, and details associated with the section name."
    "Be precise in separating content belonging to different sections."
    "If the document is a Form or a Checklist ensure to start the title with the Form or Checklist code and number then follow with its title, (eg. 'C 002 - Briefing of Masters and Senior Offiers' or 'CBO 015 - Cargo Operation Checklist')"
    "Identify and list any explicit cross-references to other documents, forms, procedures, policies, regulations, reports, their chapters and/or sections mentioned in the text."
    "Use the 'references' object in the schema, listing full names/numbers (e.g., 'Form C 002b', 'Ballast Water Management Procedure', 'SOLAS Chapter V'). Be specific and include the complete name/number as it appears or is logically inferred."
    "Adhere strictly to the provided JSON schema, including all required properties and data types."
    "Do not include trailing commas after the last item in arrays or objects."
    "CRITICAL CONTENT RULES:"
    "- Extract content as plain text only, NO formatting preservation"
    "- Replace all sequences of dots (....) with single ellipsis (...)"
    "- Replace all tabs with single space"
    "- Replace all multiple spaces with single space"
    "- Do NOT try to preserve visual alignment or formatting"
    "- Content should be clean, readable text suitable for search"
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
CONTEXT_HISTORY_WINDOW = 3  # Number of recent Q&A pairs to include in prompt

__all__ = [
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "GEMINI_SCHEMA",
    "FORM_CATEGORIES",
    "EXTRACT_SYSTEM_PROMPT",
    "CONFIDENCE_HIGH_THRESHOLD",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "MAX_CONTEXT_TURNS",
    "CONTEXT_HISTORY_WINDOW",
]