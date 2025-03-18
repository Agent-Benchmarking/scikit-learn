# Decomposition Module Example Links Progress

## Available Example Files

### Decomposition Examples
- `plot_faces_decomposition.py` - Shows various decomposition methods on face images
- `plot_pca_vs_lda.py` - Compares PCA and LDA dimensionality reduction
- `plot_sparse_coding.py` - Demonstrates sparse dictionary learning
- `plot_varimax_fa.py` - Shows factor analysis with varimax rotation
- `plot_incremental_pca.py` - Demonstrates incremental PCA
- `plot_ica_blind_source_separation.py` - Shows ICA for blind source separation
- `plot_ica_vs_pca.py` - Compares ICA and PCA
- `plot_image_denoising.py` - Image denoising with dictionary learning
- `plot_pca_iris.py` - PCA on Iris dataset
- `plot_pca_vs_fa_model_selection.py` - Compares PCA and Factor Analysis
- `plot_kernel_pca.py` - Demonstrates kernel PCA

### Cross Decomposition Examples
- `plot_compare_cross_decomposition.py` - Compares cross decomposition methods
- `plot_pcr_vs_pls.py` - Compares PCR and PLS

## Classes and Their Example Links

### PCA
- [x] Already has links to:
  - `plot_pca_iris.py`
  - `plot_pca_vs_lda.py`
  - `plot_pca_vs_fa_model_selection.py`
  - `plot_faces_decomposition.py`
  - `plot_ica_vs_pca.py`

### IncrementalPCA
- [x] Already has link to `plot_incremental_pca.py`

### KernelPCA
- [x] Already has links to:
  - `plot_kernel_pca.py`
  - `plot_digits_denoising.py`

### SparsePCA
- [x] Already has link to `plot_faces_decomposition.py`

### MiniBatchSparsePCA
- [x] Already has link to `plot_faces_decomposition.py`

### FactorAnalysis
- [x] Already has links to:
  - `plot_faces_decomposition.py`
  - `plot_varimax_fa.py`

### FastICA
- [x] Already has links to:
  - `plot_ica_blind_source_separation.py`
  - `plot_ica_vs_pca.py`
  - `plot_faces_decomposition.py`

### NMF (Non-negative Matrix Factorization)
- [x] Added links to:
  - `plot_faces_decomposition.py`
  - `plot_topics_extraction_with_nmf_lda.py`

### MiniBatchNMF
- [x] Added link to:
  - `plot_topics_extraction_with_nmf_lda.py`

### DictionaryLearning
- [x] Add link to `plot_sparse_coding.py`

### MiniBatchDictionaryLearning
- [x] Add link to `plot_image_denoising.py`

### TruncatedSVD
- [x] Already has links to `plot_document_clustering.py` and `plot_lle_digits.py`

### LatentDirichletAllocation
- [ ] Identify if any examples demonstrate LDA

## Strategy
1. First check each example file to understand what classes it demonstrates
2. Update the mapping above based on the actual content of examples
3. Add links to docstrings one by one
4. Track progress in this file

## Notes
- Focus on making examples more discoverable in the API documentation
- Will add one example reference at a time to avoid edit conflicts
