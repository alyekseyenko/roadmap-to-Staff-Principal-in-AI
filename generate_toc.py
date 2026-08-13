import re

def clean_anchor(header):
    # Lowercase
    anchor = header.lower()
    # Replace non-alphanumeric chars (excluding hyphens and spaces) with nothing
    anchor = re.sub(r'[^\w\s-]', '', anchor)
    # Replace spaces (including multiple) with a single hyphen
    anchor = re.sub(r'\s+', '-', anchor)
    return anchor

with open(r"c:\Users\habit\Desktop\lear\indice_ia_zero_a_staff_en.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

toc_lines = []
toc_lines.append("# Index — From Zero to Staff/Principal in AI\n")
toc_lines.append("<details>")
toc_lines.append("<summary><b>🗺️ Click to expand full interactive Table of Contents</b></summary>\n")

for line in lines:
    line = line.strip()
    if line.startswith("## Module"):
        # Strip the markdown prefix
        title = line.lstrip("#").strip()
        anchor = clean_anchor(title)
        toc_lines.append(f"### 📁 [{title}](#{anchor})")
    elif line.startswith("### "):
        # Strip the markdown prefix
        title = line.lstrip("#").strip()
        # Filter out anything that is not a submodule header
        if re.match(r'^[0-9.]+\s+—', title) or re.match(r'^[0-9.]+\s+-', title):
            anchor = clean_anchor(title)
            toc_lines.append(f"  * 📄 [{title}](#{anchor})")

toc_lines.append("\n</details>\n\n---")

with open(r"c:\Users\habit\Desktop\lear\scratch\toc_output.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(toc_lines))
