#!/usr/bin/env python3
import os
import sys
import requests
import json
import time

# GitHub API base URL
BASE_URL = "https://api.github.com"

# Repository details
OWNER = "Agent-Benchmarking"
REPO = "scikit-learn"
BRANCH = "feature/migrate-ubuntu-builds-to-github"

# Set GitHub token from environment variable
TOKEN = os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    print("ERROR: GITHUB_TOKEN environment variable not set.")
    print("Please set it with: export GITHUB_TOKEN=your_github_token")
    sys.exit(1)

# Headers for GitHub API requests
HEADERS = {
    "Accept": "application/vnd.github.v3+json",
    "Authorization": f"token {TOKEN}",
    "User-Agent": "workflow-trigger-script"
}

def get_workflow_id_by_name(workflow_name):
    """Get workflow ID by its name."""
    url = f"{BASE_URL}/repos/{OWNER}/{REPO}/actions/workflows"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Error getting workflows: {response.status_code}")
        print(response.json())
        return None
    
    workflows = response.json()["workflows"]
    for workflow in workflows:
        if workflow["name"] == workflow_name:
            return workflow["id"]
    
    return None

def trigger_workflow(workflow_id, workflow_name):
    """Trigger a workflow run."""
    if not workflow_id:
        print(f"Workflow '{workflow_name}' not found!")
        return False
    
    url = f"{BASE_URL}/repos/{OWNER}/{REPO}/actions/workflows/{workflow_id}/dispatches"
    payload = {
        "ref": BRANCH,
        "inputs": {}
    }
    
    response = requests.post(url, headers=HEADERS, json=payload)
    
    if response.status_code == 204:
        print(f"Successfully triggered '{workflow_name}' workflow on branch '{BRANCH}'")
        return True
    else:
        print(f"Error triggering workflow: {response.status_code}")
        try:
            print(response.json())
        except json.JSONDecodeError:
            print(f"Response: {response.text}")
        return False

def main():
    """Main function to trigger all three workflows."""
    # List of workflows to trigger
    workflows = [
        "Ubuntu Jammy Jellyfish",
        "Ubuntu Atlas",
        "Wheels"
    ]
    
    print(f"Triggering workflows for branch: {BRANCH}")
    print(f"Repository: {OWNER}/{REPO}")
    
    # Get workflow IDs
    workflow_ids = {}
    for workflow_name in workflows:
        workflow_id = get_workflow_id_by_name(workflow_name)
        if workflow_id:
            workflow_ids[workflow_name] = workflow_id
            print(f"Found workflow ID for '{workflow_name}': {workflow_id}")
        else:
            print(f"Could not find workflow ID for '{workflow_name}'")
    
    # Trigger workflows in sequence
    triggered = []
    
    # First, trigger the Jammy workflow
    if "Ubuntu Jammy Jellyfish" in workflow_ids:
        success = trigger_workflow(workflow_ids["Ubuntu Jammy Jellyfish"], "Ubuntu Jammy Jellyfish")
        if success:
            triggered.append("Ubuntu Jammy Jellyfish")
    
    # Wait a bit to ensure Jammy workflow gets registered
    if triggered:
        print("\nWaiting 30 seconds before triggering dependent workflows...")
        time.sleep(30)
    
    # Then trigger the other workflows
    for workflow_name in workflow_ids:
        if workflow_name != "Ubuntu Jammy Jellyfish":
            success = trigger_workflow(workflow_ids[workflow_name], workflow_name)
            if success:
                triggered.append(workflow_name)
    
    # Summary
    print("\nSummary:")
    for workflow_name in workflows:
        status = "Triggered" if workflow_name in triggered else "Not triggered"
        print(f"- {workflow_name}: {status}")
    
    if triggered:
        print("\nCheck workflow status with: python check_all_workflows.py")

if __name__ == "__main__":
    main() 