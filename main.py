import subprocess
import sys
import os
import urllib.request

# Install Python modules from requirements.txt
def install_python_modules():
    req_file = 'requirements.txt'
    if os.path.exists(req_file):
        print(f"Installing Python modules from {req_file}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', req_file])
    else:
        print(f"{req_file} not found. Skipping Python module installation.")

# Download GGUF model if not present
def download_gguf_model(model_dir, model_filename, url):
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_filename)
    if not os.path.exists(model_path):
        print(f"Downloading GGUF model to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
    else:
        print(f"Model {model_path} already exists. Skipping download.")

# Install llama_cpp from local wheel if present
def install_llama_cpp_wheel():
    wheel_file = 'llama_cpp_python-0.3.2-cp39-cp39-win_amd64.whl'
    if os.path.exists(wheel_file):
        print(f"Installing llama_cpp from {wheel_file}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', wheel_file])
    else:
        print(f"{wheel_file} not found. Skipping llama_cpp wheel installation.")

# run app.py using streamlit
def run_app():
    print("Running app.py using Streamlit...")
    subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', 'app.py'])

if __name__ == "__main__":
    # 1. Install llama_cpp from local wheel if present
    install_llama_cpp_wheel()

    # 2. Install Python modules
    install_python_modules()

    # 3. Download GGUF model (example: granite-3.3-2b-instruct-Q3_K_S.gguf)
    # You must provide a valid download URL for the model. This is a placeholder.
    model_dir = "Models"
    model_filename = "granite-3.3-2b-instruct-Q3_K_S.gguf"
    model_url = "https://huggingface.co/ibm-granite/granite-3.3-2b-instruct-GGUF/resolve/main/granite-3.3-2b-instruct-Q3_K_S.gguf?download=true"
    download_gguf_model(model_dir, model_filename, model_url)

    print("All dependencies are installed and model is ready.")

    # 4. Run the app
    run_app()