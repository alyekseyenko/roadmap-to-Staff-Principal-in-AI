import re

def make_safe_id(num_str, is_sub):
    # Normalize numbering like "0.0" -> "0-0", "0.0.1" -> "0-0-1"
    clean = re.sub(r'[^0-9]', '-', num_str)
    # Remove trailing/leading hyphens
    clean = clean.strip('-')
    if is_sub:
        return f"sub-{clean}"
    else:
        return f"module-{clean}"

with open(r"c:\Users\habit\Desktop\lear\indice_ia_zero_a_staff_en.md", "r", encoding="utf-8") as f:
    content = f.read()

# Locate the start of "## Module 0.0"
split_marker = "## Module 0.0 — Mathematical Foundations and Classical ML"
if "<a id=\"module-0-0\">" in content:
    # If already processed once, use the new anchor name to split
    split_marker = '## <a id="module-0-0"></a>Module 0.0 — Mathematical Foundations and Classical ML'

parts = content.split(split_marker, 1)

lines = parts[1].split("\n")
new_lines = []

toc_items = []
toc_items.append("# Index — From Zero to Staff/Principal in AI\n")
toc_items.append("<details>")
toc_items.append("<summary><b>🗺️ Click to expand full interactive Table of Contents</b></summary>\n")

# Manually insert Module 0.0 as it's our split point
toc_items.append('### 📁 [Module 0.0 — Mathematical Foundations and Classical ML](#module-0-0)')

# Process the remaining lines
for line in lines:
    stripped = line.strip()
    if stripped.startswith("## Module"):
        # Match number: "0", "1", "13"
        match = re.search(r'##\s+(?:<a id="[^"]+"></a>)?Module\s+([0-9.]+)\s*—\s*(.*)$', line)
        if match:
            num = match.group(1)
            title_text = match.group(2)
            anchor_id = make_safe_id(num, False)
            # Inject anchor
            new_line = f'## <a id="{anchor_id}"></a>Module {num} — {title_text}'
            toc_items.append(f'### 📁 [Module {num} — {title_text}](#{anchor_id})')
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    elif stripped.startswith("### "):
        # Match number: "0.0.1", "13.11"
        match = re.search(r'###\s+(?:<a id="[^"]+"></a>)?([0-9.]+)\s*—\s*(.*)$', line)
        if not match:
            # Fallback check for single hyphen
            match = re.search(r'###\s+(?:<a id="[^"]+"></a>)?([0-9.]+)\s*-\s*(.*)$', line)
            
        if match:
            num = match.group(1)
            title_text = match.group(2)
            anchor_id = make_safe_id(num, True)
            # Inject anchor
            new_line = f'### <a id="{anchor_id}"></a>{num} — {title_text}'
            toc_items.append(f'  * 📄 [{num} — {title_text}](#{anchor_id})')
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

toc_items.append("\n</details>\n\n---")

# Re-assemble everything
toc_block = "\n".join(toc_items)
document_body = "\n".join(new_lines)

final_content = toc_block + "\n\n## <a id=\"module-0-0\"></a>Module 0.0 — Mathematical Foundations and Classical ML" + document_body

with open(r"c:\Users\habit\Desktop\lear\indice_ia_zero_a_staff_en.md", "w", encoding="utf-8") as f:
    f.write(final_content)

print("Anchors injected and TOC updated successfully with Module 0.0 included!")
