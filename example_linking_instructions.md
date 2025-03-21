# Scikit-Learn Documentation Enhancement Instructions

## GitHub Issue Reference
[Issue #30621: Add links to examples to class docstrings](https://github.com/scikit-learn/scikit-learn/issues/30621)

From the issue description:
> Many scikit-learn classes don't link to examples showing them in use. This can make it harder for users to discover how to use the functionality effectively. This issue tracks adding links to appropriate examples to the class docstrings.

## Background

Scikit-Learn's documentation is improved when classes include links to relevant examples that show how to use them. These links appear at the end of docstrings under an "Examples" topic.

## Task Overview
We're working module by module, adding example links to class docstrings. Our progress is tracked in the main progress file (`example_links_progress.md`).

## Completed Modules
The following modules have been completed:
- gaussian_process.kernels - All kernel classes have been linked to appropriate examples
- kernel_approximation - Added links where appropriate examples exist
- cluster - All clustering classes have example links where appropriate
- decomposition - Added example links for specified classes
- ensemble - Added example links for all ensemble classes
- linear_model - Added example links for specified classes
- svm - Added example links for SVC, LinearSVC, SVR, and OneClassSVM

## Current In-Progress Module
- neighbors - Working on adding links to all relevant classes

## Workflow

1. **Identify Classes Needing Examples**
   - Check classes that might benefit from example links
   - Look for examples in the `examples` directory that demonstrate the class usage

2. **Match Classes to Relevant Examples**
   - Examine example files to understand what they demonstrate
   - Link classes to the most relevant examples that showcase their usage

3. **Adding Example Links**
   - For smaller files: Edit docstrings directly
   - For larger files: Use the `doc_examples_updater.py` script (located in `~/tools/`)

4. **Track Progress**
   - Update the main progress file (`example_links_progress.md`) as modules are completed
   - Create module-specific tracking files for modules currently being worked on

## Using the Automation Script

The `doc_examples_updater.py` script (located in `~/tools/`) is designed to add example links to docstrings in large files. This approach avoids timeout issues when editing large files directly.

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
   python ~/tools/doc_examples_updater.py path/to/file.py
   ```

3. Run the script for all configured files:
   ```bash
   python ~/tools/doc_examples_updater.py
   ```

## Example Link Format
```python
For [description of what the example shows] see
:ref:`sphx_glr_auto_examples_[module]_[example_filename].py`.
```

Or within an Examples topic section:
```python
.. topic:: Examples

    - :ref:`sphx_glr_auto_examples_[module]_[example_filename].py`
```

## Best Practices

1. **Search Pattern Selection**
   - Choose a unique string from the class docstring that won't be found elsewhere
   - The beginning of the docstring is typically a good choice

2. **Example Reference Format**
   - Example links should use the format: `sphx_glr_auto_examples_path_to_example.py`
   - This is transformed by Sphinx into a proper link

3. **Relevance**: Choose examples that best demonstrate the class usage
4. **Completeness**: Try to find examples for all classes in a module
5. **Consistency**: Use the same format for all example links
6. **Avoiding Duplicate Links**: The script checks for existing example sections to avoid duplication

## Common Issues and Solutions

### Line Length Limitations
- When adding example links, be mindful of the scikit-learn line length limit (88 characters)
- For long example references, break the link after `_examples_` or at another logical point:
  ```python
  - :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_text_feature_extraction
    .py`
  ```
- Our updated `doc_examples_updater.py` script now automatically handles line breaks for long example links using the following rules:
  1. For links longer than 75 characters, it breaks them after the `_examples_` part
  2. If no `_examples_` pattern is found, it breaks them at the halfway point
  3. The continuation of the link starts with 10 spaces of indentation (matching sphinx formatting)

### Docstring Pattern Matching
- If the script can't find a class docstring using the specified pattern, verify the actual docstring content
- Class docstring first lines sometimes differ from what's shown in rendered documentation
- Check the file directly to see the exact pattern to match

### Handling Pre-commit Hooks
- When committing changes, pre-commit hooks will check for code style violations
- If a commit fails due to line length issues in example links, break the lines as described above
- Run `git commit` again after fixing these issues

### Keeping doc_examples_updater.py Out of Git
- This script is a tool for our work and not part of the scikit-learn codebase
- Add it to .gitignore or be careful to not stage it for commits
- If accidentally staged, use `git restore --staged doc_examples_updater.py` to unstage it

### Identifying Recurring Issues
- If you identify recurring issues not mentioned in these instructions, update this documentation
- Add this instruction about updating instructions to the instructions as well

## Repository Organization
- Main progress file: `example_links_progress.md` - Tracks overall progress across all modules
- Current module progress file: `neighbors_links_progress.md` - Tracks detailed progress on the neighbors module
- Helper script: `doc_examples_updater.py` - Located in `~/tools/` (not tracked in the repository)

## Commit and Push Changes

After updating documentation:
1. Test that the examples are correctly linked by building the documentation locally (if possible)
2. **Review your changes before committing**:
   ```bash
   git diff
   ```
3. Commit your changes:
   ```bash
   git add path/to/updated/files
   git commit -m "DOC Add example links to [class names]"
   ```
4. Push your changes:
   ```bash
   git push origin your-branch-name
   ```
