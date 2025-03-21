# Scikit-Learn Documentation Enhancement Instructions

This document provides instructions on how to enhance Scikit-Learn docstrings by adding example links.

## Background

Scikit-Learn's documentation is improved when classes include links to relevant examples that show how to use them. These links appear at the end of docstrings under a "Examples" topic.

## Workflow

1. **Identify Classes Needing Examples**
   - Check classes that might benefit from example links
   - Look for examples in the `examples` directory that demonstrate the class usage

2. **Match Classes to Relevant Examples**
   - Examine example files to understand what they demonstrate
   - Link classes to the most relevant examples that showcase their usage

3. **Adding Example Links**
   - For smaller files: You can edit docstrings directly
   - For larger files: Use the `doc_examples_updater.py` script to avoid timeout issues

4. **Track Progress**
   - Maintain a progress file (like `linear_model_links_progress.md`) to track which classes have been updated
   - Document which examples were linked to each class

## Using the Automation Script

The `doc_examples_updater.py` script is designed to add example links to docstrings in large files. This approach avoids timeout issues when editing large files directly.

### How to Use the Script

1. Add your class configuration to the `class_configs` dictionary:
   ```python
   "YourClass": {
       "file_path": "path/to/file.py",
       "search_pattern": "Unique string from class docstring",
       "example_links": [
           "sphx_glr_auto_examples_path_to_example.py"
       ],
       "end_pattern": '"""'
   }
   ```

2. Run the script for a specific file:
   ```bash
   python doc_examples_updater.py path/to/file.py
   ```

3. Run the script for all configured files:
   ```bash
   python doc_examples_updater.py
   ```

## Important Notes

1. **Search Pattern Selection**
   - Choose a unique string from the class docstring that won't be found elsewhere
   - The beginning of the docstring is typically a good choice

2. **Example Reference Format**
   - Example links should use the format: `sphx_glr_auto_examples_path_to_example.py`
   - This is transformed by Sphinx into a proper link

3. **Tracking Progress**
   - Keep your progress file updated as you add links
   - Mark completed classes with [x] and note which examples were linked

4. **Avoiding Duplicate Links**
   - The script checks for existing example sections to avoid duplication
   - If a class already has examples, it will be skipped

## Best Practices

1. **Relevance**: Choose examples that best demonstrate the class usage
2. **Completeness**: Try to find examples for all classes in a module
3. **Consistency**: Use the same format for all example links
4. **Documentation**: Keep progress tracking up-to-date

## Commit and Push Changes

After updating documentation:
1. Test that the examples are correctly linked by building the documentation locally (if possible)
2. **Review your changes before committing**:
   ```bash
   git diff
   ```
   This is a crucial step to verify all changes are as expected before committing.
3. Commit your changes:
   ```bash
   git add path/to/updated/files
   git commit -m "DOC Add example links to [class names]"
   ```
   **Important:** Never use the `--no-verify` flag when committing. Allow the pre-commit hooks to run as they help catch errors and maintain code quality standards.
4. Push your changes:
   ```bash
   git push origin your-branch-name
   ```
