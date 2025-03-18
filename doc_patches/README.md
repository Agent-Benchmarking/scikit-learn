# Documentation Patch Files

This directory contains patch files for adding example links to various scikit-learn linear model classes. These patches were created because direct edits to the larger files were experiencing timeout issues.

## Patch Files

The following patch files have been created:

1. `elasticnet_patch.txt` - Adds example links to the ElasticNet class docstring
2. `elasticnetcv_patch.txt` - Adds example links to the ElasticNetCV class docstring
3. `lasso_patch.txt` - Adds example links to the Lasso class docstring
4. `lassocv_patch.txt` - Adds example links to the LassoCV class docstring
5. `lassolars_patch.txt` - Adds example links to the LassoLars class docstring
6. `lassolars_cv_patch.txt` - Adds example links to the LassoLarsCV class docstring
7. `lassolarsic_patch.txt` - Adds example links to the LassoLarsIC class docstring
8. `multitasklasso_patch.txt` - Adds example links to the MultiTaskLasso class docstring

## How to Apply Patches

To apply these patches, you can use the `patch` command. Navigate to the root directory of the scikit-learn repository and run:

```bash
# For a single patch
patch -p0 < doc_patches/elasticnet_patch.txt

# Or to apply all patches at once
for patch_file in doc_patches/*.txt; do
    patch -p0 < "$patch_file"
done
```

Note: If you encounter any issues with the patches, you can try applying them with the `-p1` option or manually edit the files based on the content in the patch files.

## Tracking Progress

The progress of documentation enhancements is tracked in the `linear_model_links_progress.md` file. Classes marked with `[-]` have patch files created but not yet applied to the codebase.
