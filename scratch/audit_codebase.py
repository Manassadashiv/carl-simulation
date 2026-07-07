import os
import hashlib
import ast
import re

ROOT_DIR = r"D:\carl_simulation"
REPORT_PATH = r"C:\Users\MANAS\AppData\Local\Temp\codebase_audit_report.md" # We will write it to the brain directory instead.
# Let's find the correct brain directory. Wait, we can get it from python's os.environ or pass it, or write it directly to D:\carl_simulation\codebase_audit_report.md.
# Let's write the report to D:\carl_simulation\codebase_audit_report.md first, then we can copy it or copy its content.
REPORT_PATH = r"D:\carl_simulation\codebase_audit_report.md"

EXCLUDE_DIRS = {
    'venv',
    '__pycache__',
    '.git',
    'mujoco_menagerie',
    'archive',
    'logs',
    'brain',
    'dashboard',
    'memory',
    'final_build_checkpoint'
}

KEYWORDS = ['todo', 'fixme', 'stub', 'dummy', 'placeholder', 'notimplemented', 'xxx', 'temp', 'blank', 'misleading']

def get_hash(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def check_ast(content, filename):
    try:
        tree = ast.parse(content, filename=filename)
    except Exception as e:
        return [f"AST parsing error: {e}"]
    
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            is_empty = False
            body = node.body
            reason = ""
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    is_empty = True
                    reason = "pass statement only"
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    is_empty = True
                    reason = "docstring only"
                elif isinstance(stmt, ast.Raise):
                    exc = stmt.exc
                    if isinstance(exc, ast.Call) and getattr(exc.func, 'id', '') == 'NotImplementedError':
                        is_empty = True
                        reason = "raises NotImplementedError"
                    elif isinstance(exc, ast.Name) and exc.id == 'NotImplementedError':
                        is_empty = True
                        reason = "raises NotImplementedError"
                elif isinstance(stmt, ast.Return) and stmt.value is None:
                    is_empty = True
                    reason = "empty return only"
            if is_empty:
                findings.append(f"Line {node.lineno}: Empty function/method `{node.name}` ({reason})")
        elif isinstance(node, ast.ClassDef):
            is_empty = False
            body = node.body
            reason = ""
            if len(body) == 1:
                stmt = body[0]
                if isinstance(stmt, ast.Pass):
                    is_empty = True
                    reason = "pass statement only"
                elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    is_empty = True
                    reason = "docstring only"
            if is_empty:
                findings.append(f"Line {node.lineno}: Empty class `{node.name}` ({reason})")
    return findings

def audit():
    file_list = []
    for root, dirs, files in os.walk(ROOT_DIR):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in ['.py', '.xml', '.txt', '.html']:
                file_list.append(os.path.join(root, file))

    hashes = {}
    empty_files = []
    keyword_matches = []
    ast_empty_nodes = {}
    
    for path in file_list:
        rel_path = os.path.relpath(path, ROOT_DIR)
        size = os.path.getsize(path)
        
        if size == 0:
            empty_files.append(rel_path)
            continue
            
        file_hash = get_hash(path)
        if file_hash:
            hashes.setdefault(file_hash, []).append(rel_path)
            
        # Read file content
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {rel_path}: {e}")
            continue
            
        # Check keywords
        lines = content.splitlines()
        for idx, line in enumerate(lines, 1):
            for kw in KEYWORDS:
                # search for whole word or sub-word but clean
                match = re.search(r'\b' + re.escape(kw) + r'\b', line, re.IGNORECASE)
                if match:
                    # Ignore comment blocks that are benign if any, but let's list them
                    # Check if it's in a comment or string
                    keyword_matches.append({
                        'file': rel_path,
                        'line': idx,
                        'keyword': kw,
                        'text': line.strip()
                    })
                    
        # AST analysis for python files
        if path.endswith('.py'):
            ast_findings = check_ast(content, path)
            if ast_findings:
                ast_empty_nodes[rel_path] = ast_findings

    # Process duplicates
    duplicates = {}
    for h, paths in hashes.items():
        if len(paths) > 1:
            duplicates[paths[0]] = paths[1:]

    # Write Report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("# Codebase Audit Report\n\n")
        f.write("This report lists potential loose ends, dummy code, blank files, misleading markers, and file redundancies across the codebase.\n\n")
        
        f.write("## 1. Empty Files (0 Bytes)\n")
        if empty_files:
            for ef in empty_files:
                f.write(f"- `{ef}`\n")
        else:
            f.write("No empty files found.\n")
        f.write("\n")
        
        f.write("## 2. File Redundancy (Identical Duplicates)\n")
        f.write("Files in `GENESIS_BIPED` and `GENESIS_HAND` that are identical to their counterparts in `GENESIS` or other directories:\n\n")
        if duplicates:
            # Group by directories
            for primary, secondary in sorted(duplicates.items()):
                f.write(f"- Primary: `{primary}`\n")
                for sec in sorted(secondary):
                    f.write(f"  - Duplicate: `{sec}`\n")
        else:
            f.write("No exact duplicates found.\n")
        f.write("\n")
        
        f.write("## 3. Empty Python Classes & Functions (AST Analysis)\n")
        f.write("Functions or classes that have empty or stubbed bodies (e.g., only contain `pass`, docstring, or raise `NotImplementedError`):\n\n")
        if ast_empty_nodes:
            for file, findings in sorted(ast_empty_nodes.items()):
                f.write(f"### `{file}`\n")
                for finding in findings:
                    f.write(f"- {finding}\n")
                f.write("\n")
        else:
            f.write("No empty classes/functions found.\n")
        f.write("\n")
        
        f.write("## 4. Incomplete Work Keywords (TODO, FIXME, STUB, DUMMY, etc.)\n")
        f.write("Instances of development keywords found in code comments or docstrings:\n\n")
        if keyword_matches:
            # Group by file
            grouped = {}
            for m in keyword_matches:
                grouped.setdefault(m['file'], []).append(m)
                
            for file, matches in sorted(grouped.items()):
                f.write(f"### `{file}`\n")
                for m in matches:
                    f.write(f"- **Line {m['line']}** (`{m['keyword'].upper()}`): `{m['text']}`\n")
                f.write("\n")
        else:
            f.write("No keyword matches found.\n")
            
    print(f"Audit completed. Report written to {REPORT_PATH}")

if __name__ == '__main__':
    audit()
