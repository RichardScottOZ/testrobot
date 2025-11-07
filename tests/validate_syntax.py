"""
Syntax Validation Script

Validates Python syntax of all framework files without requiring dependencies.
"""

import py_compile
import os
import sys
from pathlib import Path


def check_syntax(file_path):
    """Check syntax of a Python file."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)


def main():
    """Main validation function."""
    print("Validating Python syntax for Multimodal Semantic Search Framework")
    print("=" * 70)
    
    # Find all Python files
    root_dir = Path(__file__).parent.parent
    python_files = []
    
    # Add specific files and directories
    patterns = [
        'semantic_search/**/*.py',
        '*.py',
        'examples/*.py',
        'tests/*.py'
    ]
    
    for pattern in patterns:
        python_files.extend(root_dir.glob(pattern))
    
    # Remove duplicates and sort
    python_files = sorted(set(python_files))
    
    # Check each file
    errors = []
    success_count = 0
    
    for file_path in python_files:
        relative_path = file_path.relative_to(root_dir)
        print(f"Checking {relative_path}...", end=" ")
        
        success, error = check_syntax(str(file_path))
        
        if success:
            print("✓ OK")
            success_count += 1
        else:
            print("✗ FAIL")
            errors.append((relative_path, error))
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Results: {success_count}/{len(python_files)} files passed")
    
    if errors:
        print("\nErrors found:")
        for path, error in errors:
            print(f"\n  {path}:")
            print(f"    {error}")
        return False
    else:
        print("\n✓ All Python files have valid syntax!")
        return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
