import re
from typing import List, Tuple

# Robust section heading regex:
# Matches optional numbering (e.g., "1", "1.1"), followed by common section keywords
SECTION_HDR = re.compile(
    r"^(?:\d+\.?\d*\s*)?(ABSTRACT|INTRODUCTION|METHODS?|EXPERIMENTS?|RESULTS?|DISCUSSION|CONCLUSION)\b",
    re.IGNORECASE | re.MULTILINE
)


def split_into_sections(text: str) -> List[Tuple[str, str]]:
    """
    Splits a document into (section_name, section_text) pairs using robust heading detection.
    Falls back to a single FULL_TEXT section if no headings are found.

    :param text: Full document text
    :return: List of tuples: (SECTION_NAME, SECTION_BODY)
    """
    # Find all headings with positions
    matches = [(m.start(), m.group(1).upper()) for m in SECTION_HDR.finditer(text)]

    # If no headings found, return entire text as FULL_TEXT
    if not matches:
        return [("FULL_TEXT", text.strip())]

    sections: List[Tuple[str, str]] = []
    # Handle preamble before first heading
    first_pos, _ = matches[0]
    if first_pos > 0:
        pre_text = text[:first_pos].strip()
        if pre_text:
            sections.append(("PREFACE", pre_text))

    # Iterate through found headings
    for idx, (pos, name) in enumerate(matches):
        # Determine start of section body: after the heading line
        newline_idx = text.find("\n", pos)
        body_start = newline_idx + 1 if newline_idx != -1 else pos
        # Determine end: next heading pos or end of text
        end = matches[idx+1][0] if idx+1 < len(matches) else len(text)
        section_text = text[body_start:end].strip()
        sections.append((name, section_text))

    return sections


# if __name__ == "__main__":
#     # Quick standalone test
#     sample = """
#     1 Introduction
#     This is the intro.
#     2 Methods
#     Here are the methods.
#     3 Results
#     Results go here.
#     """
#     for sec, body in split_into_sections(sample):
#         print(f"=== {sec} ===")
#         print(body)
#         print()
