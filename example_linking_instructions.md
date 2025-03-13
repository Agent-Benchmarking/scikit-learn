# Instructions for Adding Example Links to scikit-learn Documentation

## GitHub Issue Reference
[Issue #30621: Add links to examples to class docstrings](https://github.com/scikit-learn/scikit-learn/issues/30621)

From the issue description:
> Many scikit-learn classes don't link to examples showing them in use. This can make it harder for users to discover how to use the functionality effectively. This issue tracks adding links to appropriate examples to the class docstrings.

## Task Overview
The task is to add links to existing examples in the scikit-learn documentation for various classes. We're working module by module, currently focusing on:
1. Gaussian process kernels (completed)
2. Kernel approximation (completed)
3. Clustering (in progress)

## Progress Tracking
- Main progress file: `/home/augment/Desktop/Code/scikit-learn/example_links_progress.md`
- Module-specific progress files:
  - `/home/augment/Desktop/Code/scikit-learn/kernel_links_progress.md`
  - `/home/augment/Desktop/Code/scikit-learn/kernel_links_progress_detailed.md`
  - `/home/augment/Desktop/Code/scikit-learn/kernel_approximation_links_progress.md`
  - `/home/augment/Desktop/Code/scikit-learn/cluster_links_progress.md`

Always consult these files to see what has been done and what remains to be done.

## Systematic Approach

### For Each Module:
1. Create a tracking file if it doesn't exist
2. Map available examples to appropriate classes
3. Work through classes one by one
4. Update tracking files as progress is made

### For Each Class:
1. Check the progress tracking file to identify which examples need to be added
2. View the class docstring to understand its current structure
3. View the example files to understand their content
4. Add example links to the class docstring after existing examples
5. Mark completed tasks in the tracking file

### Example Link Format:
```python
For [description of what the example shows] see
:ref:`sphx_glr_auto_examples_[module]_[example_filename].py`.
```

## Troubleshooting
If the edit_file tool encounters errors:
1. Use view_file to check the current state of the file
2. Try adding one example link at a time
3. Use more specific line ranges
4. Check for syntax issues in the edit

## Next Actions
1. Continue adding example links according to the progress files
2. Keep tracking files up to date
3. Move to the next module once current module is completed

Remember to check each docstring and example carefully to ensure the description accurately reflects what the example demonstrates.
