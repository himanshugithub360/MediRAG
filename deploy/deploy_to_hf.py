"""Deploy the app to a free Hugging Face Space (Docker SDK).

Run by GitHub Actions on every push to main (see .github/workflows/ci.yml), or by hand:

    set HF_TOKEN=hf_...            (a Hugging Face access token with "write" permission)
    set HF_SPACE_ID=your-username/medirag
    python deploy/deploy_to_hf.py

It creates the Space if needed, stores the API keys as Space *secrets* (if they are set in
the environment), and uploads only the files the app needs. Never .env, Data/ or chat history.
"""
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Only these are uploaded (an allow-list, so secrets and local data can't leak)
FILES = ["app.py", "wsgi.py", "serve.py", "Dockerfile", ".dockerignore", "requirements.txt", "setup.py"]
FOLDERS = ["src", "templates", "static"]
SECRETS = ["PINECONE_API_KEY", "MISTRAL_API_KEY", "FLASK_SECRET_KEY"]
VARIABLES = {
    "TRUST_PROXY": "1",          # real visitor IPs for rate limiting behind the HF proxy
    "CROSS_SITE_COOKIES": "1",   # the Space page shows the app in an iframe from *.hf.space
}


def main():
    token = os.environ.get("HF_TOKEN")
    space_id = os.environ.get("HF_SPACE_ID")
    if not token or not space_id:
        sys.exit("Set HF_TOKEN and HF_SPACE_ID (e.g. your-username/medirag) first.")

    api = HfApi(token=token)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{space_id}")

    for name in SECRETS:
        if os.environ.get(name):
            api.add_space_secret(space_id, name, os.environ[name])
            print(f"  secret {name} set")
        else:
            print(f"  secret {name} not provided (set it in the Space settings if it's missing)")
    for name, value in VARIABLES.items():
        api.add_space_variable(space_id, name, value)

    with tempfile.TemporaryDirectory() as staging:
        for name in FILES:
            shutil.copy2(os.path.join(ROOT, name), staging)
        for folder in FOLDERS:
            shutil.copytree(os.path.join(ROOT, folder), os.path.join(staging, folder),
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # The Space's README holds its settings (front matter) and a short description
        shutil.copy2(os.path.join(ROOT, "deploy", "huggingface", "README.md"), os.path.join(staging, "README.md"))

        try:
            sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            sha = "local"
        api.upload_folder(
            folder_path=staging, repo_id=space_id, repo_type="space",
            commit_message=f"Deploy {sha} from GitHub",
            # remove files that were deleted from the project since the last deploy
            delete_patterns=["*.py", "src/*", "templates/*", "static/*"],
        )
    print("Uploaded. The Space rebuilds automatically; the first start takes about 5-10 minutes.")
    print(f"App URL: https://{space_id.replace('/', '-').replace('_', '-').lower()}.hf.space")


if __name__ == "__main__":
    main()
