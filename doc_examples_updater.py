#!/usr/bin/env python3
"""
Script to update scikit-learn docstrings with example links.
This approach allows for precise targeting of docstrings in large files
where direct edits may time out.
"""

import sys
import re


def add_example_links(
    file_path, class_name, search_pattern, example_links, end_pattern=None
):
    """
    Add example links to a docstring at a specific location in a file.
    
    Parameters
    ----------
    file_path : str
        Path to the file containing the docstring to update
    class_name : str
        Name of the class being updated (for logging purposes)
    search_pattern : str
        Pattern to search for to locate the insertion point
    example_links : list of str
        List of example links to add to the docstring
    end_pattern : str, optional
        Pattern marking the end of the docstring, used to avoid 
        adding duplicate links
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the insertion point
    index = content.find(search_pattern)
    if index == -1:
        print(
            f"[{class_name}] Pattern '{search_pattern}' not found in {file_path}"
        )
        return False

    # Set insertion point to end of the search pattern
    insert_index = index + len(search_pattern)

    # Check if we need to move to a specific ending pattern
    if end_pattern:
        end_index = content.find(end_pattern, insert_index)
        if end_index == -1:
            print(
                f"[{class_name}] End pattern '{end_pattern}' not found after search pattern in {file_path}"
            )
            return False
        insert_index = end_index

    # Check if links are already present (to avoid duplication)
    examples_section = "    .. topic:: Examples:"
    # Check a reasonable portion after insertion point
    links_snippet = content[insert_index : insert_index + 500]

    if examples_section in links_snippet:
        print(
            f"[{class_name}] Example links already appear to be present at insertion point in {file_path}"
        )
        return False

    # Create links block
    links_block = "\n\n    .. topic:: Examples:\n\n"
    for link in example_links:
        links_block += f"        - :ref:`{link}`\n"

    # Insert the links block before the docstring end
    new_content = content[:insert_index] + links_block + content[insert_index:]

    # Write back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[{class_name}] Successfully added example links to {file_path}")
    return True


if __name__ == "__main__":
    # Configuration for each class
    coordinate_descent_path = "sklearn/linear_model/_coordinate_descent.py"
    least_angle_path = "sklearn/linear_model/_least_angle.py"
    docstring_end = '"""'
    
    # Dictionary of class configurations
    class_configs = {
        # Coordinate Descent Models
        "ElasticNet": {
            "file_path": coordinate_descent_path,
            "search_pattern": "Linear regression with combined L1 and L2 priors as regularizer.",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py",
                "sphx_glr_auto_examples_linear_model_plot_elastic_net_precomputed_gram_matrix_with_weighted_samples.py",
                "sphx_glr_auto_examples_inspection_plot_train_error_vs_test_error.py"
            ],
            "end_pattern": docstring_end
        },
        "Lasso": {
            "file_path": coordinate_descent_path,
            "search_pattern": "Linear Model trained with L1 prior as regularizer (aka the Lasso).",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_and_elasticnet.py",
                "sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py",
                "sphx_glr_auto_examples_linear_model_plot_lasso_lars_ic.py",
                "sphx_glr_auto_examples_applications_plot_tomography_l1_reconstruction.py"
            ],
            "end_pattern": docstring_end
        },
        "LassoCV": {
            "file_path": coordinate_descent_path,
            "search_pattern": "Lasso linear model with iterative fitting along a regularization path.",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py",
                "sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py"
            ],
            "end_pattern": docstring_end
        },
        "ElasticNetCV": {
            "file_path": coordinate_descent_path,
            "search_pattern": "Elastic Net model with iterative fitting along a regularization path.",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py"
            ],
            "end_pattern": docstring_end
        },
        "MultiTaskLasso": {
            "file_path": coordinate_descent_path,
            "search_pattern": "Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_multi_task_lasso_support.py"
            ],
            "end_pattern": docstring_end
        },
        
        # Least Angle Regression Models
        "LassoLars": {
            "file_path": least_angle_path,
            "search_pattern": "Lasso model fit with Least Angle Regression a.k.a. Lars",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_lasso_lars_elasticnet_path.py"
            ],
            "end_pattern": docstring_end
        },
        "LassoLarsCV": {
            "file_path": least_angle_path,
            "search_pattern": "Cross-validated Lasso, using the LARS algorithm",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py"
            ],
            "end_pattern": docstring_end
        },
        "LassoLarsIC": {
            "file_path": least_angle_path,
            "search_pattern": "Lasso model fit with Lars using BIC or AIC for model selection",
            "example_links": [
                "sphx_glr_auto_examples_linear_model_plot_lasso_lars_ic.py",
                "sphx_glr_auto_examples_linear_model_plot_lasso_model_selection.py"
            ],
            "end_pattern": docstring_end
        }
    }
    
    # Process each class configuration
    target_files = sys.argv[1:] if len(sys.argv) > 1 else list(set(config["file_path"] for config in class_configs.values()))
    success_count = 0
    processed_count = 0
    
    for class_name, config in class_configs.items():
        # Skip classes not in target files
        if config["file_path"] not in target_files:
            continue
            
        processed_count += 1
        result = add_example_links(
            config["file_path"],
            class_name,
            config["search_pattern"],
            config["example_links"],
            config["end_pattern"]
        )
        if result:
            success_count += 1
    
    print(
        f"\nUpdate summary: Successfully updated {success_count} out of {processed_count} classes"
    )
    
    # Exit with status code 0 if at least one class was updated successfully
    sys.exit(0 if success_count > 0 else 1)
