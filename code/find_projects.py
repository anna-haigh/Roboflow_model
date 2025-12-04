#!/usr/bin/env python3
"""
Find available Roboflow projects and datasets
"""

import requests
import json


def list_workspace_projects(api_key):
    """List all projects in the workspace"""
    
    print("="*50)
    print("FINDING YOUR ROBOFLOW PROJECTS")
    print("="*50)
    
    # Get workspace info
    print("\nFetching workspace information...")
    url = "https://api.roboflow.com/"
    params = {"api_key": api_key}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    
    workspace = data.get('workspace', {})
    workspace_name = workspace.get('name', 'Unknown')
    workspace_url = workspace.get('url', 'Unknown')
    
    print(f"\nWorkspace: {workspace_name}")
    print(f"Workspace ID: {workspace_url}")
    print("\n" + "="*50)
    print("AVAILABLE PROJECTS")
    print("="*50)
    
    projects = workspace.get('projects', [])
    
    if not projects:
        print("No projects found in this workspace!")
        return
    
    for i, project in enumerate(projects, 1):
        project_name = project.get('name', 'Unknown')
        project_id = project.get('id', 'Unknown')
        
        print(f"\n{i}. Project Name: {project_name}")
        print(f"   Project ID: {project_id}")
        
        # Try to get versions
        try:
            versions_url = f"https://api.roboflow.com/{workspace_url}/{project_id}"
            versions_response = requests.get(versions_url, params=params)
            
            if versions_response.status_code == 200:
                versions_data = versions_response.json()
                versions = versions_data.get('versions', [])
                
                if versions:
                    print(f"   Available Versions:")
                    for version in versions:
                        version_num = version.get('id', '?')
                        version_name = version.get('name', 'Unnamed')
                        print(f"     - Version {version_num}: {version_name}")
                else:
                    print(f"   No versions found")
        except Exception as e:
            print(f"   Could not fetch versions: {e}")
    
    print("\n" + "="*50)
    print("\nUSAGE:")
    print("To download annotations, use:")
    print(f"  Workspace ID: {workspace_url}")
    print(f"  Project ID: [one of the project IDs above]")
    print(f"  Version: [version number from above]")
    print("="*50)


def main():
    print("\nROBOFLOW PROJECT FINDER")
    print("="*50)
    
    api_key = input("Enter your Roboflow API key: ").strip()
    
    if not api_key:
        print("Error: API key required!")
        return
    
    list_workspace_projects(api_key)


if __name__ == "__main__":
    main()
