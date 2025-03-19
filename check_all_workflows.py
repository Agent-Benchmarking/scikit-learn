#!/usr/bin/env python3
import os
import sys
import requests
import time
from datetime import datetime

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
    "User-Agent": "workflow-check-script"
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

def format_time(timestamp):
    """Format GitHub timestamp to a readable format."""
    if not timestamp:
        return "N/A"
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def get_latest_workflow_run(workflow_id):
    """Get the latest workflow run for a specific workflow."""
    if not workflow_id:
        return None
    
    url = f"{BASE_URL}/repos/{OWNER}/{REPO}/actions/workflows/{workflow_id}/runs"
    params = {
        "branch": BRANCH,
        "per_page": 1
    }
    
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code != 200:
        print(f"Error getting workflow runs: {response.status_code}")
        print(response.json())
        return None
    
    runs = response.json()["workflow_runs"]
    return runs[0] if runs else None

def get_workflow_jobs(run_id):
    """Get jobs for a specific workflow run."""
    url = f"{BASE_URL}/repos/{OWNER}/{REPO}/actions/runs/{run_id}/jobs"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Error getting workflow jobs: {response.status_code}")
        print(response.json())
        return []
    
    return response.json()["jobs"]

def print_job_details(jobs):
    """Print details of all jobs in a workflow run."""
    for job in jobs:
        status = job["status"]
        conclusion = job["conclusion"] if job["status"] == "completed" else "N/A"
        step_failures = []
        
        # Check for failed steps
        for step in job.get("steps", []):
            if step.get("conclusion") == "failure":
                step_failures.append(step["name"])
        
        print(f"  - Job: {job['name']}")
        print(f"    Status: {status}")
        print(f"    Conclusion: {conclusion}")
        print(f"    Started: {format_time(job.get('started_at'))}")
        print(f"    Completed: {format_time(job.get('completed_at'))}")
        print(f"    URL: {job['html_url']}")
        
        if step_failures:
            print("    Failed steps:")
            for step_name in step_failures:
                print(f"      - {step_name}")
        print()

def check_workflow(workflow_name):
    """Check the status of a specific workflow."""
    print(f"\n=== Checking {workflow_name} Workflow ===")
    
    workflow_id = get_workflow_id_by_name(workflow_name)
    if not workflow_id:
        print(f"Workflow '{workflow_name}' not found!")
        return
    
    latest_run = get_latest_workflow_run(workflow_id)
    if not latest_run:
        print(f"No runs found for '{workflow_name}' on branch '{BRANCH}'")
        return
    
    # Print workflow run details
    run_id = latest_run["id"]
    status = latest_run["status"]
    conclusion = latest_run["conclusion"] if latest_run["status"] == "completed" else "N/A"
    url = latest_run["html_url"]
    
    print(f"Run ID: {run_id}")
    print(f"Status: {status}")
    print(f"Conclusion: {conclusion}")
    print(f"Created: {format_time(latest_run.get('created_at'))}")
    print(f"Updated: {format_time(latest_run.get('updated_at'))}")
    print(f"URL: {url}")
    
    # Get and print job details
    print("\nJobs:")
    jobs = get_workflow_jobs(run_id)
    print_job_details(jobs)

def main():
    """Main function to check all three workflows."""
    # List of workflows to check
    workflows = [
        "Ubuntu Jammy Jellyfish",
        "Ubuntu Atlas",
        "Wheels"
    ]
    
    print(f"Checking workflows for branch: {BRANCH}")
    print(f"Repository: {OWNER}/{REPO}")
    
    for workflow_name in workflows:
        check_workflow(workflow_name)
        # Add a small delay between requests to avoid rate limiting
        time.sleep(1)

if __name__ == "__main__":
    main() 