import re

with open('phase10b_world.py', encoding='utf-8') as f:
    content = f.read()

def fix_docstring(m):
    inner = m.group(1).strip().replace('\n', ' ')
    while '  ' in inner:
        inner = inner.replace('  ', ' ')
    return f'    # {inner}'

content = re.sub(r'    """(.*?)"""', fix_docstring, content, flags=re.DOTALL)

with open('phase10b_world.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('DONE')
