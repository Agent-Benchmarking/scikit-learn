"""Get the list of selected tests for the current build."""

import os
import re
import sys

# Add the parent directory to the path so we can import get_commit_message
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from github.get_commit_message import get_commit_message


def get_selected_tests():
    """Get the list of selected tests for the current build.

    Returns
    -------
    selected_tests : str
        The list of selected tests, or an empty string if no tests are selected.
    """
    commit_message = get_commit_message()

    # Look for a pattern like [test sklearn/test_file.py::test_func]
    match = re.search(r"\[test (.*)\]", commit_message)
    if match:
        return match.group(1)

    return ""


if __name__ == "__main__":
    selected_tests = get_selected_tests()
    if selected_tests:
        print(f"Selected tests: {selected_tests}")
        os.environ["SELECTED_TESTS"] = selected_tests
    else:
        print("No tests selected.")
        os.environ["SELECTED_TESTS"] = ""
