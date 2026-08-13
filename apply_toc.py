# Read TOC content
with open(r"c:\Users\habit\Desktop\lear\scratch\toc_output.txt", "r", encoding="utf-8") as f:
    toc = f.read()

# Read target file content
with open(r"c:\Users\habit\Desktop\lear\indice_ia_zero_a_staff_en.md", "r", encoding="utf-8") as f:
    content = f.read()

# Locate the start of "## Module 0.0"
split_marker = "## Module 0.0 — Mathematical Foundations and Classical ML"
parts = content.split(split_marker, 1)

# Re-assemble with TOC first, followed by the rest of the document
new_content = toc + "\n\n" + split_marker + parts[1]

# Save back to target file
with open(r"c:\Users\habit\Desktop\lear\indice_ia_zero_a_staff_en.md", "w", encoding="utf-8") as f:
    f.write(new_content)

print("TOC successfully merged!")
