import argparse
import json
from pathlib import Path

from app.config import AppConfig
from app.extraction import gemini_extract_record

# Initialize config
config = AppConfig.get()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug Gemini extraction with a specific PDF file."
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF to inspect.",
    )
    return parser.parse_args()


def resolve_pdf_path(raw_path: str) -> Path:
    pdf_path = Path(raw_path).expanduser()
    if pdf_path.is_absolute():
        pdf_path = pdf_path.resolve()
    else:
        pdf_path = (Path.cwd() / pdf_path).resolve()

    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found: {pdf_path}")

    return pdf_path


def main() -> None:
    args = parse_args()
    pdf_path = resolve_pdf_path(args.pdf_path)

    print(f"Extracting: {pdf_path.name}")
    print("=" * 60)

    # Run extraction
    result = gemini_extract_record(pdf_path)

    print("\n" + "=" * 60)
    print("EXTRACTION RESULTS:")
    print("=" * 60)

    print(f"Doc Type: {result.get('doc_type')}")
    print(f"Title: {result.get('title')}")
    print(f"Sections: {len(result.get('sections', []))}")
    print(f"Parse Error: {result.get('parse_error', 'None')}")
    print(f"Multi-pass: {result.get('multi_pass_extraction', False)}")

    if result.get("sections"):
        print("\nFirst 5 sections:")
        for i, section in enumerate(result["sections"][:5], 1):
            print(f"  {i}. {section.get('name', 'UNNAMED')}")

    # Check for tables in a specific section
    echo_section = next(
        (s for s in result.get("sections", []) if "Echo Sounder" in s.get("name", "")),
        None,
    )
    if echo_section:
        content = echo_section["content"]
        has_table = "|" in content or "Responsible" in content
        print(
            f"\nEcho Sounder section: {len(content)} chars, "
            f"has table markers: {has_table}"
        )

    # Save full result for inspection
    with open("debug_pdf_result.json", "w") as f:
        json.dump(result, f, indent=2)
        print("\nFull result saved to debug_pdf_result.json")


if __name__ == "__main__":
    main()
