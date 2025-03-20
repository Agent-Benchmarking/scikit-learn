"""Get the commit message for the current build."""

import os
import subprocess
import sys


def get_commit_message():
    """Get the commit message for the current build.

    Returns
    -------
    commit_message : str
        The commit message.
    """
    # Try to get the commit message from Azure Pipelines
    if "BUILD_SOURCEVERSIONMESSAGE" in os.environ:
        return os.environ["BUILD_SOURCEVERSIONMESSAGE"]

    # Try to get the commit message from GitHub Actions
    if "GITHUB_EVENT_NAME" in os.environ:
        if os.environ["GITHUB_EVENT_NAME"] == "pull_request":
            # For pull requests, use the last commit message in the PR
            cmd = ["git", "log", "-1", "--pretty=%B"]
        else:
            # For pushes, use the last non-merge commit message
            cmd = ["git", "log", "--no-merges", "-1", "--pretty=%B"]

        try:
            return subprocess.check_output(cmd, universal_newlines=True).strip()
        except subprocess.CalledProcessError:
            print("Failed to get commit message from git log")

    # Default to empty string if we can't get the commit message
    return ""


def main():
    """Print the commit message."""
    commit_message = get_commit_message()
    print(commit_message)

    # For Azure Pipelines, set the output variable
    if "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI" in os.environ:
        print(f"##vso[task.setvariable variable=message;isOutput=true]{commit_message}")


if __name__ == "__main__":
    # If --only-show-message is passed, only print the message
    if len(sys.argv) > 1 and sys.argv[1] == "--only-show-message":
        print(get_commit_message())
    else:
        main()
