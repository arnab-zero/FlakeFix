"""
FlakeFix Backend - GitHub Webhook Receiver
Minimal FastAPI backend to receive GitHub webhooks
"""

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
import logging
import os
from dotenv import load_dotenv
import httpx
import tempfile
import subprocess
import shutil
from pathlib import Path
import re
import json
import asyncio
import uuid
from datetime import datetime

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="FlakeFix Backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GitHub token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    logger.warning("GITHUB_TOKEN not set - API calls will fail")

# GitHub OAuth credentials
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Repository storage directory
REPO_STORAGE_DIR = os.getenv("REPO_STORAGE_DIR", "G:/FlakeFix-Temporary-Repo-Storage")
os.makedirs(REPO_STORAGE_DIR, exist_ok=True)
logger.info(f"Repository storage directory: {REPO_STORAGE_DIR}")

# In-memory progress tracking
analysis_progress = {}


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "FlakeFix Backend is running"}


@app.get("/auth/github")
async def github_auth():
    """
    Redirect to GitHub OAuth authorization page
    """
    if not GITHUB_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")
    
    github_auth_url = (
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&redirect_uri=http://localhost:8000/auth/github/callback"
        f"&scope=repo,user"
    )
    
    return RedirectResponse(url=github_auth_url)


@app.get("/auth/github/callback")
async def github_callback(code: str):
    """
    Handle GitHub OAuth callback and exchange code for access token
    """
    if not GITHUB_CLIENT_ID or not GITHUB_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")
    
    try:
        # Exchange code for access token
        async with httpx.AsyncClient() as client:
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                headers={"Accept": "application/json"},
                data={
                    "client_id": GITHUB_CLIENT_ID,
                    "client_secret": GITHUB_CLIENT_SECRET,
                    "code": code,
                }
            )
            token_response.raise_for_status()
            token_data = token_response.json()
            
            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="Failed to get access token")
            
            # Get user info
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Accept": "application/json"
                }
            )
            user_response.raise_for_status()
            user_data = user_response.json()
            
            # Redirect back to frontend with token and user data
            redirect_url = (
                f"{FRONTEND_URL}/auth/callback"
                f"?token={access_token}"
                f"&user={user_data['login']}"
                f"&avatar={user_data['avatar_url']}"
            )
            
            return RedirectResponse(url=redirect_url)
            
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        return RedirectResponse(url=f"{FRONTEND_URL}?error=auth_failed")


@app.post("/analyze")
async def analyze_repository(request: Request, background_tasks: BackgroundTasks):
    """
    Manual repository analysis endpoint (triggered from frontend form)
    """
    try:
        # Get request data
        data = await request.json()
        
        owner = data.get("owner")
        repo = data.get("repo")
        branch = data.get("branch", "main")
        commit_message = data.get("commitMessage", "")
        
        logger.info(f"Manual analysis request: {owner}/{repo} (branch: {branch})")
        
        if not owner or not repo:
            raise HTTPException(status_code=400, detail="Missing owner or repo")
        
        repo_full_name = f"{owner}/{repo}"
        
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Clone repository
        # For manual analysis, we'll clone the latest commit from the specified branch
        # We need to get the latest commit SHA first
        async with httpx.AsyncClient() as client:
            # Get branch info to get latest commit SHA
            branch_url = f"https://api.github.com/repos/{repo_full_name}/branches/{branch}"
            headers = {
                "Authorization": f"token {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            response = await client.get(branch_url, headers=headers)
            response.raise_for_status()
            branch_data = response.json()
            commit_sha = branch_data["commit"]["sha"]
            
            logger.info(f"Latest commit on {branch}: {commit_sha}")
        
        # Clone repository
        repo_path = await clone_repository(repo_full_name, commit_sha)
        
        if not repo_path:
            raise HTTPException(status_code=500, detail="Failed to clone repository")
        
        logger.info(f"Repository cloned to: {repo_path}")
        
        # Find all test files in the repository
        test_files = []
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if is_test_file(file):
                    # Get relative path from repo root
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, repo_path)
                    test_files.append(rel_path)
        
        logger.info(f"Found {len(test_files)} test files")
        
        # Parse test methods from all test files
        test_methods = []
        for test_file in test_files:
            file_path = os.path.join(repo_path, test_file)
            methods = parse_test_methods(file_path)
            
            if methods:
                logger.info(f"Found {len(methods)} test methods in {test_file}")
                for method in methods:
                    test_methods.append({
                        'file': test_file,
                        'method': method
                    })
        
        logger.info(f"Total test methods to analyze: {len(test_methods)}")
        
        if not test_methods:
            return {
                "status": "success",
                "message": "No test methods found in repository",
                "test_methods_count": 0,
                "analysis_id": analysis_id
            }
        
        # Initialize progress tracking
        analysis_progress[analysis_id] = {
            "status": "initializing",
            "repository": repo_full_name,
            "branch": branch,
            "commit_sha": commit_sha,
            "currentTest": 0,
            "totalTests": len(test_methods),
            "tests": [
                {
                    "file": t["file"],
                    "method": t["method"],
                    "status": "pending",
                    "steps": {
                        "call_graph": "pending",
                        "inlining": "pending",
                        "token_counting": "pending",
                        "detection": "pending",
                        "patch": "pending"
                    },
                    "result": None,
                    "error": None
                }
                for t in test_methods
            ],
            "started_at": datetime.now().isoformat()
        }
        
        # Run pipeline in background
        background_tasks.add_task(
            run_pipeline_for_manual_analysis,
            analysis_id,
            repo_path,
            repo_full_name,
            test_methods
        )
        
        logger.info(f"Pipeline execution queued in background (ID: {analysis_id})")
        
        return {
            "status": "success",
            "message": "Analysis started",
            "analysis_id": analysis_id,
            "repository": repo_full_name,
            "branch": branch,
            "commit_sha": commit_sha,
            "test_methods_count": len(test_methods)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{analysis_id}/progress")
async def get_analysis_progress(analysis_id: str):
    """
    Get progress of an ongoing analysis
    """
    if analysis_id not in analysis_progress:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    return analysis_progress[analysis_id]


@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    Receive GitHub webhook events and extract changed test files
    """
    try:
        # Get the payload
        payload = await request.json()
        
        # Log webhook received
        event_type = request.headers.get("X-GitHub-Event", "unknown")
        logger.info(f"Received GitHub webhook: {event_type}")
        
        # Only process pull_request events
        if event_type != "pull_request":
            logger.info(f"Ignoring event type: {event_type}")
            return {"status": "ignored", "reason": "Not a pull_request event"}
        
        # Get action
        action = payload.get("action", "")
        logger.info(f"Action: {action}")
        
        # Only process opened, synchronize (new commits), reopened
        if action not in ["opened", "synchronize", "reopened"]:
            logger.info(f"Ignoring action: {action}")
            return {"status": "ignored", "reason": f"Action '{action}' not relevant"}
        
        # Get PR details
        pr = payload["pull_request"]
        pr_number = pr["number"]
        pr_title = pr["title"]
        repo_full_name = payload["repository"]["full_name"]
        
        logger.info(f"Processing PR #{pr_number}: {pr_title}")
        logger.info(f"Repository: {repo_full_name}")
        
        # Get changed files from PR
        changed_files = await get_changed_files(payload)
        
        # Filter for test files only
        test_files = [f for f in changed_files if is_test_file(f)]
        
        logger.info(f"Total changed files: {len(changed_files)}")
        logger.info(f"Test files changed: {len(test_files)}")
        
        if test_files:
            logger.info("Changed test files:")
            for test_file in test_files:
                logger.info(f"  - {test_file}")
            
            # Clone repository and get local path
            repo_path = await clone_repository(repo_full_name, pr["head"]["sha"])
            
            if repo_path:
                logger.info(f"Repository cloned to: {repo_path}")
                
                # Parse test methods from changed test files
                test_methods = []
                for test_file in test_files:
                    file_path = os.path.join(repo_path, test_file)
                    methods = parse_test_methods(file_path)
                    
                    if methods:
                        logger.info(f"Found {len(methods)} test methods in {test_file}:")
                        for method in methods:
                            logger.info(f"  - {method}")
                            test_methods.append({
                                'file': test_file,
                                'method': method
                            })
                    else:
                        logger.warning(f"No test methods found in {test_file}")
                
                logger.info(f"Total test methods to analyze: {len(test_methods)}")
                
                # Run pipeline in background for each test method
                if test_methods:
                    logger.info("About to queue background task...")
                    background_tasks.add_task(
                        run_pipeline_for_tests,
                        repo_path,
                        repo_full_name,
                        test_methods,
                        pr_number
                    )
                    logger.info("Pipeline execution queued in background")
                    logger.info("Background task should start processing now...")
                
                # Note: Repository is kept in storage directory, not deleted
            else:
                logger.error("Failed to clone repository")
        else:
            logger.info("No test files changed")
        
        return {
            "status": "processed",
            "pr_number": pr_number,
            "repository": repo_full_name,
            "total_files_changed": len(changed_files),
            "test_files_changed": len(test_files),
            "test_files": test_files
        }
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_changed_files(payload: dict) -> list:
    """
    Fetch list of changed files from GitHub API
    """
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set - cannot fetch changed files")
        return []
    
    try:
        # Extract PR details
        repo_full_name = payload["repository"]["full_name"]
        pr_number = payload["pull_request"]["number"]
        
        # GitHub API endpoint for PR files
        url = f"https://api.github.com/repos/{repo_full_name}/pulls/{pr_number}/files"
        
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Make API request
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            
            files_data = response.json()
            
            # Extract filenames
            changed_files = [file["filename"] for file in files_data]
            
            return changed_files
            
    except Exception as e:
        logger.error(f"Error fetching changed files from GitHub API: {e}")
        return []


def is_test_file(filename: str) -> bool:
    """
    Check if a file is a Java test file
    """
    return filename.endswith("Test.java") or filename.endswith("Tests.java")


async def clone_repository(repo_full_name: str, commit_sha: str) -> str:
    """
    Clone GitHub repository to storage directory
    Returns the path to the cloned repository, or None if failed
    """
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN not set - cannot clone repository")
        return None
    
    try:
        # Create directory for this repo (e.g., G:/FlakeFix-Temporary-Repo-Storage/owner_repo_commit)
        repo_name = repo_full_name.replace("/", "_")
        commit_short = commit_sha[:7]
        repo_dir = os.path.join(REPO_STORAGE_DIR, f"{repo_name}_{commit_short}")
        
        # If already exists, use it
        if os.path.exists(repo_dir):
            logger.info(f"Repository already exists: {repo_dir}")
            return repo_dir
        
        logger.info(f"Creating directory: {repo_dir}")
        os.makedirs(repo_dir, exist_ok=True)
        
        # Clone URL with token
        clone_url = f"https://{GITHUB_TOKEN}@github.com/{repo_full_name}.git"
        
        # Clone repository (shallow clone for speed)
        logger.info(f"Cloning repository: {repo_full_name}")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, repo_dir],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            shutil.rmtree(repo_dir, ignore_errors=True)
            return None
        
        # Checkout specific commit
        logger.info(f"Checking out commit: {commit_sha}")
        result = subprocess.run(
            ["git", "fetch", "origin", commit_sha],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            subprocess.run(
                ["git", "checkout", commit_sha],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
        else:
            logger.warning(f"Could not checkout specific commit, using default branch")
        
        logger.info(f"Repository cloned successfully to: {repo_dir}")
        return repo_dir
        
    except subprocess.TimeoutExpired:
        logger.error("Git clone timeout")
        if repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        return None
    except Exception as e:
        logger.error(f"Error cloning repository: {e}")
        if repo_dir and os.path.exists(repo_dir):
            shutil.rmtree(repo_dir, ignore_errors=True)
        return None


def cleanup_repository(repo_path: str):
    """
    Delete the cloned repository directory
    """
    try:
        if repo_path and os.path.exists(repo_path):
            shutil.rmtree(repo_path, ignore_errors=True)
            logger.info(f"Cleaned up repository: {repo_path}")
    except Exception as e:
        logger.error(f"Error cleaning up repository: {e}")


def parse_test_methods(file_path: str) -> list:
    """
    Parse Java test file to extract test method names
    Looks for methods annotated with @Test
    """
    test_methods = []
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return test_methods
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Pattern to match @Test annotation followed by method declaration
        # Matches: @Test ... public void methodName(...) or @Test ... void methodName(...)
        pattern = r'@Test\s+(?:public\s+)?(?:void|[A-Z]\w*)\s+(\w+)\s*\('
        
        matches = re.findall(pattern, content, re.MULTILINE)
        test_methods = list(set(matches))  # Remove duplicates
        
        logger.debug(f"Parsed {len(test_methods)} test methods from {file_path}")
        
    except Exception as e:
        logger.error(f"Error parsing test methods from {file_path}: {e}")
    
    return test_methods


def run_pipeline_for_manual_analysis(analysis_id: str, repo_path: str, repo_full_name: str, test_methods: list):
    """
    Run the flaky detection pipeline for manual analysis (no PR posting)
    This runs in the background
    """
    try:
        logger.info("="*70)
        logger.info(f"MANUAL ANALYSIS BACKGROUND TASK STARTED (ID: {analysis_id})")
        logger.info(f"Starting pipeline execution for {len(test_methods)} test methods")
        logger.info(f"Repository: {repo_full_name}")
        logger.info(f"Repo Path: {repo_path}")
        logger.info("="*70)
        
        # Update status to running
        if analysis_id in analysis_progress:
            analysis_progress[analysis_id]["status"] = "running"
        
        results = []
        project_name = repo_full_name.split('/')[1]  # Extract repo name from owner/repo
        
        for idx, test_info in enumerate(test_methods, 1):
            test_file = test_info['file']
            test_method = test_info['method']
            
            logger.info(f"\n[{idx}/{len(test_methods)}] Running pipeline for {test_file}::{test_method}")
            
            # Update progress: mark current test as in_progress
            if analysis_id in analysis_progress:
                analysis_progress[analysis_id]["currentTest"] = idx
                analysis_progress[analysis_id]["tests"][idx-1]["status"] = "in_progress"
                analysis_progress[analysis_id]["tests"][idx-1]["steps"]["call_graph"] = "in_progress"
            
            try:
                # Get absolute path to run_pipeline.py (one level up from backend/)
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(backend_dir)
                run_pipeline_script = os.path.join(project_root, "run_pipeline.py")
                
                logger.info(f"Backend dir: {backend_dir}")
                logger.info(f"Project root: {project_root}")
                logger.info(f"Pipeline script: {run_pipeline_script}")
                logger.info(f"Script exists: {os.path.exists(run_pipeline_script)}")
                
                # Simulate step progress (since we can't track real-time from subprocess)
                # In a real implementation, you'd parse pipeline output or use a message queue
                if analysis_id in analysis_progress:
                    analysis_progress[analysis_id]["tests"][idx-1]["steps"]["call_graph"] = "completed"
                    analysis_progress[analysis_id]["tests"][idx-1]["steps"]["inlining"] = "in_progress"
                
                # Run the pipeline
                cmd = [
                    "python",
                    run_pipeline_script,
                    "--project-name", project_name,
                    "--project-root", repo_path,
                    "--test-file", test_file,
                    "--test-method", test_method,
                    "--output-dir", "output"
                ]
                logger.info(f"Executing command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per test
                    cwd=project_root,  # Run from project root
                    encoding='utf-8',  # Force UTF-8 encoding to handle Unicode characters
                    errors='replace'  # Replace unencodable characters instead of failing
                )
                
                logger.info(f"Pipeline process completed with return code: {result.returncode}")
                
                # Log pipeline output
                if result.stdout:
                    logger.info(f"Pipeline stdout for {test_method}:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Pipeline stderr for {test_method}:\n{result.stderr}")
                
                if result.returncode == 0:
                    logger.info(f"Pipeline completed successfully for {test_method}")
                    
                    # Read results
                    results_file = os.path.join(
                        project_root,
                        "output",
                        f"{project_name}_{test_method}_results",
                        "results.json"
                    )
                    
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            pipeline_result = json.load(f)
                            results.append({
                                'test_file': test_file,
                                'test_method': test_method,
                                'status': 'success',
                                'result': pipeline_result
                            })
                            
                            # Update progress with results
                            if analysis_id in analysis_progress:
                                analysis_progress[analysis_id]["tests"][idx-1]["status"] = "completed"
                                analysis_progress[analysis_id]["tests"][idx-1]["steps"] = {
                                    "call_graph": "completed",
                                    "inlining": "completed",
                                    "token_counting": "completed",
                                    "detection": "completed",
                                    "patch": "completed"
                                }
                                analysis_progress[analysis_id]["tests"][idx-1]["result"] = {
                                    "is_flaky": pipeline_result.get("summary", {}).get("is_flaky", False),
                                    "label": pipeline_result.get("summary", {}).get("label", ""),
                                    "confidence": pipeline_result.get("summary", {}).get("confidence", ""),
                                    "patch_generated": pipeline_result.get("summary", {}).get("patch_generated", False)
                                }
                    else:
                        logger.warning(f"Results file not found: {results_file}")
                        results.append({
                            'test_file': test_file,
                            'test_method': test_method,
                            'status': 'error',
                            'error': 'Results file not found'
                        })
                        
                        if analysis_id in analysis_progress:
                            analysis_progress[analysis_id]["tests"][idx-1]["status"] = "failed"
                            analysis_progress[analysis_id]["tests"][idx-1]["error"] = "Results file not found"
                else:
                    logger.error(f"Pipeline failed for {test_method}: {result.stderr}")
                    results.append({
                        'test_file': test_file,
                        'test_method': test_method,
                        'status': 'error',
                        'error': result.stderr
                    })
                    
                    if analysis_id in analysis_progress:
                        analysis_progress[analysis_id]["tests"][idx-1]["status"] = "failed"
                        analysis_progress[analysis_id]["tests"][idx-1]["error"] = result.stderr[:200]
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Pipeline timeout for {test_method}")
                results.append({
                    'test_file': test_file,
                    'test_method': test_method,
                    'status': 'error',
                    'error': 'Pipeline timeout (10 minutes)'
                })
                
                if analysis_id in analysis_progress:
                    analysis_progress[analysis_id]["tests"][idx-1]["status"] = "failed"
                    analysis_progress[analysis_id]["tests"][idx-1]["error"] = "Timeout (10 minutes)"
            except Exception as e:
                logger.error(f"Error running pipeline for {test_method}: {e}")
                results.append({
                    'test_file': test_file,
                    'test_method': test_method,
                    'status': 'error',
                    'error': str(e)
                })
                
                if analysis_id in analysis_progress:
                    analysis_progress[analysis_id]["tests"][idx-1]["status"] = "failed"
                    analysis_progress[analysis_id]["tests"][idx-1]["error"] = str(e)[:200]
        
        logger.info(f"Manual analysis completed for all {len(test_methods)} test methods")
        logger.info(f"Results: {len([r for r in results if r['status'] == 'success'])} successful, "
                    f"{len([r for r in results if r['status'] == 'error'])} failed")
        logger.info(f"Results saved to output/ directory")
        
        # Mark analysis as completed
        if analysis_id in analysis_progress:
            analysis_progress[analysis_id]["status"] = "completed"
            analysis_progress[analysis_id]["completed_at"] = datetime.now().isoformat()
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in manual analysis background task: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Mark analysis as failed
        if analysis_id in analysis_progress:
            analysis_progress[analysis_id]["status"] = "failed"
            analysis_progress[analysis_id]["error"] = str(e)


def run_pipeline_for_tests(repo_path: str, repo_full_name: str, test_methods: list, pr_number: int):
    """
    Run the flaky detection pipeline for each test method
    This runs in the background
    """
    try:
        logger.info("="*70)
        logger.info(f"BACKGROUND TASK STARTED")
        logger.info(f"Starting pipeline execution for {len(test_methods)} test methods")
        logger.info(f"Repository: {repo_full_name}")
        logger.info(f"PR Number: {pr_number}")
        logger.info(f"Repo Path: {repo_path}")
        logger.info("="*70)
        
        results = []
        project_name = repo_full_name.split('/')[1]  # Extract repo name from owner/repo
        
        for idx, test_info in enumerate(test_methods, 1):
            test_file = test_info['file']
            test_method = test_info['method']
            
            logger.info(f"\n[{idx}/{len(test_methods)}] Running pipeline for {test_file}::{test_method}")
            
            try:
                # Get absolute path to run_pipeline.py (one level up from backend/)
                backend_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(backend_dir)
                run_pipeline_script = os.path.join(project_root, "run_pipeline.py")
                
                logger.info(f"Backend dir: {backend_dir}")
                logger.info(f"Project root: {project_root}")
                logger.info(f"Pipeline script: {run_pipeline_script}")
                logger.info(f"Script exists: {os.path.exists(run_pipeline_script)}")
                
                # Run the pipeline
                cmd = [
                    "python",
                    run_pipeline_script,
                    "--project-name", project_name,
                    "--project-root", repo_path,
                    "--test-file", test_file,
                    "--test-method", test_method,
                    "--output-dir", "output"
                ]
                logger.info(f"Executing command: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout per test
                    cwd=project_root,  # Run from project root
                    encoding='utf-8',  # Force UTF-8 encoding to handle Unicode characters
                    errors='replace'  # Replace unencodable characters instead of failing
                )
                
                logger.info(f"Pipeline process completed with return code: {result.returncode}")
                
                # Log pipeline output
                if result.stdout:
                    logger.info(f"Pipeline stdout for {test_method}:\n{result.stdout}")
                if result.stderr:
                    logger.warning(f"Pipeline stderr for {test_method}:\n{result.stderr}")
                
                if result.returncode == 0:
                    logger.info(f"Pipeline completed successfully for {test_method}")
                    
                    # Read results
                    results_file = os.path.join(
                        project_root,
                        "output",
                        f"{project_name}_{test_method}_results",
                        "results.json"
                    )
                    
                    if os.path.exists(results_file):
                        with open(results_file, 'r') as f:
                            pipeline_result = json.load(f)
                            results.append({
                                'test_file': test_file,
                                'test_method': test_method,
                                'status': 'success',
                                'result': pipeline_result
                            })
                    else:
                        logger.warning(f"Results file not found: {results_file}")
                        results.append({
                            'test_file': test_file,
                            'test_method': test_method,
                            'status': 'error',
                            'error': 'Results file not found'
                        })
                else:
                    logger.error(f"Pipeline failed for {test_method}: {result.stderr}")
                    results.append({
                        'test_file': test_file,
                        'test_method': test_method,
                        'status': 'error',
                        'error': result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                logger.error(f"Pipeline timeout for {test_method}")
                results.append({
                    'test_file': test_file,
                    'test_method': test_method,
                    'status': 'error',
                    'error': 'Pipeline timeout (10 minutes)'
                })
            except Exception as e:
                logger.error(f"Error running pipeline for {test_method}: {e}")
                results.append({
                    'test_file': test_file,
                    'test_method': test_method,
                    'status': 'error',
                    'error': str(e)
                })
        
        logger.info(f"Pipeline execution completed for all {len(test_methods)} test methods")
        logger.info(f"Results: {len([r for r in results if r['status'] == 'success'])} successful, "
                    f"{len([r for r in results if r['status'] == 'error'])} failed")
        
        # Post results to GitHub PR (run in event loop)
        asyncio.run(post_results_to_github(repo_full_name, pr_number, results))
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in background task: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


async def post_results_to_github(repo_full_name: str, pr_number: int, results: list):
    """
    Post analysis results as a comment on the GitHub PR
    """
    try:
        logger.info(f"Posting results to GitHub PR #{pr_number}")
        
        # Format results as markdown comment
        comment = format_results_comment(results)
        
        # Post comment via GitHub API
        url = f"https://api.github.com/repos/{repo_full_name}/issues/{pr_number}/comments"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json={"body": comment},
                timeout=30
            )
            response.raise_for_status()
        
        logger.info(f"Successfully posted results to PR #{pr_number}")
        
    except Exception as e:
        logger.error(f"Failed to post results to GitHub: {e}")


def format_results_comment(results: list) -> str:
    """
    Format analysis results as a markdown comment for GitHub PR
    """
    # Count results
    total = len(results)
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'error']
    flaky_tests = [r for r in successful if r.get('result', {}).get('summary', {}).get('is_flaky')]
    
    # Build comment
    comment = "## 🔍 FlakeFix Analysis Results\n\n"
    comment += f"**Tests Analyzed:** {total}\n"
    comment += f"**Flaky Tests Found:** {len(flaky_tests)}\n"
    comment += f"**Analysis Failures:** {len(failed)}\n\n"
    
    if flaky_tests:
        comment += "---\n\n"
        comment += "### 🚨 Flaky Tests Detected\n\n"
        
        for result in flaky_tests:
            summary = result['result']['summary']
            test_file = result['test_file']
            test_method = result['test_method']
            
            comment += f"#### `{test_file}::{test_method}`\n\n"
            comment += f"- **Type:** {summary['label']}\n"
            comment += f"- **Confidence:** {summary['confidence']}\n"
            comment += f"- **Detection Method:** {summary['detection_method']}\n"
            
            # Add detection details if available
            detection_result = result['result']['steps'].get('detection', {})
            if 'zero_shot_result' in detection_result:
                zero_shot = detection_result['zero_shot_result']
                comment += f"\n**Issue:**\n{zero_shot.get('justification', 'N/A')}\n"
                
                if 'recommendation' in zero_shot:
                    comment += f"\n**Recommendation:**\n{zero_shot['recommendation']}\n"
            
            # Add patch info
            if summary.get('patch_generated'):
                comment += f"\n**Patch:** ✅ Generated successfully\n"
                patch_file = result['result']['steps']['patch'].get('output_file')
                if patch_file:
                    comment += f"- Location: `{patch_file}`\n"
            else:
                comment += f"\n**Patch:** ❌ Not generated\n"
            
            comment += "\n---\n\n"
    
    if not flaky_tests and not failed:
        comment += "### ✅ No Flaky Tests Detected\n\n"
        comment += "All analyzed tests appear to be stable.\n\n"
    
    if failed:
        comment += "### ⚠️ Analysis Failures\n\n"
        for result in failed:
            test_file = result['test_file']
            test_method = result['test_method']
            comment += f"- `{test_file}::{test_method}` - Analysis failed\n"
        comment += "\n"
    
    comment += "---\n\n"
    comment += "*Powered by [FlakeFix](https://github.com/your-org/flakefix)*"
    
    return comment


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
