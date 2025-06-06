# InnovateX-IBM-Hackathon-2025-Smart-Internz

# SmartSDLC AI Assistant

SmartSDLC AI Assistant is a Streamlit-based application that leverages local LLMs (Llama.cpp) to assist with various software development lifecycle (SDLC) tasks, including SDLC phase generation, project abstract writing, code review, documentation generation, bug analysis, and project chat. The app is designed for privacy and offline use, running entirely on your machine.

![image](https://github.com/user-attachments/assets/9227ccce-5361-4fab-a8c5-3ebbb0e4a3b5)

## Features
- **Project Management:** Create, select, and manage multiple software projects.
- **AI-Powered SDLC Generation:** Automatically generate SDLC outlines for your projects.
- **Abstract Generator:** Write professional project abstracts with AI.
- **Code Review Assistant:** Get AI-powered code review and suggestions.
- **Documentation Assistant:** Generate technical documentation and docstrings.
- **Bug Detection & Resolution:** Analyze code or error messages and get AI-suggested fixes.
- **Chat with AI:** Ask project-related questions and get instant answers.
- **User Preferences:** Fine-tune AI parameters (temperature, max tokens, top-p, top-k, repetition penalty, GPU layers).
- **Local Model Execution:** Uses `llama-cpp-python` and a local GGUF model for privacy and speed.

## Setup Instructions

Run the provided script to install all required dependencies, download the model and run the app:

```powershell
python main.py
```

This will:
- Install packages from `requirements.txt`
- Install `llama_cpp_python` from the local wheel (if present)
- Download the GGUF model to the `Models/` directory
- Run the app.py

## Usage
- Open the app in your browser (Streamlit will provide a local URL).
- Create a new project or select an existing one.
- Use the various AI tools to assist with your SDLC tasks.
- All data and models remain local for privacy.

## Model & Requirements
- The app uses a local GGUF model (e.g., `granite-3.3-2b-instruct-Q3_K_S.gguf`).
- You can change the model by updating the model file in the `Models/` directory and editing the path in `app.py`.
- All dependencies are listed in `requirements.txt`.

## File Structure
- `app.py` - Main Streamlit app
- `install_dependencies.py` - Dependency/model installer
- `requirements.txt` - Python dependencies
- `llama_cpp_python-0.3.2-cp39-cp39-win_amd64.whl` - Local wheel for llama_cpp
- `Models/` - Directory for GGUF models
- `AI_Tasks_Data/` - Stores task state (pending, running, finished)
- `Projects_Data_and_User_Preferences/` - Stores project and user preference data

## Notes
- The app is designed for Windows and Python 3.9+.
- Ensure you have enough RAM and disk space for the model.
- For best performance, use a machine with a compatible GPU (or set GPU layers to 0 for CPU-only).
