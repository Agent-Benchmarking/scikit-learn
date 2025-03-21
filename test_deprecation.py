import warnings

import numpy as np

from sklearn.utils import _safe_indexing

# Test that using _safe_indexing raises a deprecation warning
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _safe_indexing(np.array([[1, 2], [3, 4]]), 0)
    assert len(w) == 1
    assert issubclass(w[0].category, FutureWarning)
    assert "'_safe_indexing' is deprecated" in str(w[0].message)
