import os

file_path = r'd:\carl_simulation\phase16_alife.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for idx, line in enumerate(lines):
    # From line 604 onwards, outdent by 4 spaces up to line 952 (1-indexed)
    if 603 <= idx <= 951:
        if line.startswith('    '):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    elif idx == 602:
        new_lines.append("                continue\n") # ensure line 603 is correct
    else:
        new_lines.append(line)

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Indentation fixed.")
