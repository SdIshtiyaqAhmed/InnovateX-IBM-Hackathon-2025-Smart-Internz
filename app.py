import streamlit as st
import uuid
import json
import os
from llama_cpp import Llama
import time

# --- Streamlit UI Setup ---
st.set_page_config(page_title="SmartSDLC AI Assistant", layout="wide")
st.session_state.queue_processed_this_run = False

# --- File Paths ---
PREFERENCES_FILE = "./Projects_Data_and_User_Preferences/user_preferences.json"
PROJECTS_FILE = "./Projects_Data_and_User_Preferences/projects_data.json"
AI_PENDING_TASKS_FILE = "./AI_Tasks_Data/ai_pending_tasks.json" # New file for pending tasks
AI_RUNNING_TASK_FILE = "./AI_Tasks_Data/ai_running_task.json"   # New file for the single running task
AI_FINISHED_TASKS_FILE = "./AI_Tasks_Data/ai_finished_tasks.json" # New file for finished tasks (replaces history)

# Ensure directories exist for all files
for file_path in [PREFERENCES_FILE, PROJECTS_FILE, AI_PENDING_TASKS_FILE, AI_RUNNING_TASK_FILE, AI_FINISHED_TASKS_FILE]:
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

# --- Default User Preferences ---
DEFAULT_PREFERENCES = {
    "ai_temperature": 0.7,
    "ai_max_new_tokens": 8192,
    "ai_top_p": 0.9,
    "ai_top_k": 50,
    "ai_repetition_penalty": 1.1,
    "ai_n_gpu_layers": 0
}

# --- Preference Management Functions ---
def load_preferences():
    if os.path.exists(PREFERENCES_FILE):
        with open(PREFERENCES_FILE, 'r') as f:
            try:
                prefs = json.load(f)
                return {**DEFAULT_PREFERENCES, **prefs}
            except json.JSONDecodeError:
                st.warning("Could not decode preferences file. Using default settings.")
                return DEFAULT_PREFERENCES
    return DEFAULT_PREFERENCES

def save_preferences(preferences):
    with open(PREFERENCES_FILE, 'w') as f:
        json.dump(preferences, f, indent=4)

# --- Project Management Functions ---
def load_projects():
    if os.path.exists(PROJECTS_FILE):
        try:
            with open(PROJECTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            st.warning("Could not load saved projects. Starting with empty projects.")
    return {}

def save_projects(projects):
    try:
        with open(PROJECTS_FILE, "w") as f:
            json.dump(projects, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save projects: {e}")

# --- AI Task Queue Persistence (New Functions) ---
def load_ai_pending_tasks():
    if os.path.exists(AI_PENDING_TASKS_FILE):
        try:
            with open(AI_PENDING_TASKS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            st.warning("Could not load saved AI pending tasks. Starting with empty queue.")
    return []

def save_ai_pending_tasks(tasks):
    try:
        with open(AI_PENDING_TASKS_FILE, "w") as f:
            json.dump(tasks, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save AI pending tasks: {e}")

def load_ai_running_task():
    if os.path.exists(AI_RUNNING_TASK_FILE):
        try:
            with open(AI_RUNNING_TASK_FILE, "r") as f:
                # Should always be a list, even if it contains 0 or 1 task
                running_tasks = json.load(f)
                return running_tasks[0] if running_tasks else None
        except Exception:
            st.warning("Could not load saved AI running task. Starting with no running task.")
    return None

def save_ai_running_task(task):
    try:
        with open(AI_RUNNING_TASK_FILE, "w") as f:
            # Always save as a list, even if it's empty or contains one task
            json.dump([task] if task else [], f, indent=4)
    except Exception as e:
        st.error(f"Failed to save AI running task: {e}")

def load_ai_finished_tasks():
    if os.path.exists(AI_FINISHED_TASKS_FILE):
        try:
            with open(AI_FINISHED_TASKS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            st.warning("Could not load saved AI finished tasks. Starting with empty history.")
    return []

def save_ai_finished_tasks(history):
    try:
        with open(AI_FINISHED_TASKS_FILE, "w") as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save AI finished tasks: {e}")

# --- Session State Initialization ---
if "current_project_id" not in st.session_state:
    st.session_state.current_project_id = None
if "user_preferences" not in st.session_state:
    st.session_state.user_preferences = load_preferences()
if "projects" not in st.session_state:
    st.session_state.projects = load_projects()

# Initialize task queues/history and clear on startup for pending/running
if "app_initialized" not in st.session_state:
    st.session_state.app_initialized = True
    # Clear pending and running tasks at the start of the program
    save_ai_pending_tasks([])
    save_ai_running_task(None)
    st.session_state.ai_pending_tasks = []
    st.session_state.ai_running_task = None
    st.session_state.ai_finished_tasks = load_ai_finished_tasks() # Load finished history
else:
    st.session_state.ai_pending_tasks = load_ai_pending_tasks()
    st.session_state.ai_running_task = load_ai_running_task()
    st.session_state.ai_finished_tasks = load_ai_finished_tasks()


if "code_review_status" not in st.session_state:
    st.session_state.code_review_status = "not_started"
if "doc_input" not in st.session_state:
    st.session_state.doc_input = ""
if "doc_result" not in st.session_state:
    st.session_state.doc_result = ""
if "doc_status" not in st.session_state:
    st.session_state.doc_status = "not_started"
if "bug_input" not in st.session_state:
    st.session_state.bug_input = ""
if "bug_result" not in st.session_state:
    st.session_state.bug_result = ""
if "bug_status" not in st.session_state:
    st.session_state.bug_status = "not_started"
if "abstract_gen_project_name_input" not in st.session_state:
    st.session_state.abstract_gen_project_name_input = ""
if "abstract_gen_description_input" not in st.session_state:
    st.session_state.abstract_gen_description_input = ""
if "abstract_gen_result" not in st.session_state:
    st.session_state.abstract_gen_result = ""
if "abstract_gen_status" not in st.session_state:
    st.session_state.abstract_gen_status = "not_started"
if "sdlc_status" not in st.session_state:
    st.session_state.sdlc_status = "not_started"
if "code_input_review" not in st.session_state:
    st.session_state.code_input_review = ""
# New session states for displaying full-width output
if "display_task_output_id" not in st.session_state:
    st.session_state.display_task_output_id = None
if "display_task_output_content" not in st.session_state:
    st.session_state.display_task_output_content = ""
if "display_task_output_title" not in st.session_state:
    st.session_state.display_task_output_title = ""


# --- Helper: get_default_project_data ---
def get_default_project_data(project_name="New Project", project_abstract="", sdlc_phases_content=None):
    return {
        "name": project_name,
        "abstract": project_abstract,
        "chat_history": [],
        "sdlc_phases_content": sdlc_phases_content,
    }

# --- Helper: add_ai_task (Modified) ---
def add_ai_task(task_type, input_text, project_name=None, tool_label=None):
    new_task = {
        'id': str(uuid.uuid4()),
        'type': task_type,
        'input': input_text,
        'project_name': project_name,
        'tool_label': tool_label,
        'status': 'pending',
        'result': None,
        'error': None,
        'timestamp': time.time()
    }
    pending_tasks = st.session_state.get("ai_pending_tasks", [])
    pending_tasks.append(new_task)
    st.session_state.ai_pending_tasks = pending_tasks
    save_ai_pending_tasks(pending_tasks)
    # Set flag to hide tools immediately after submit
    st.session_state.ai_task_just_submitted = True
    st.rerun()

# --- Helper: on_preference_change ---
def on_preference_change():
    save_preferences(st.session_state.user_preferences)

# --- Helper: get_task_ai_settings ---
def get_task_ai_settings(task_type, user_preferences):
    return user_preferences.copy()

# --- Model Loading (Cached for Performance) ---
GGUF_MODEL_PATH = "./Models/granite-3.3-2b-instruct-Q3_K_S.gguf"
@st.cache_resource
def load_granite_model_cached(gguf_model_path, n_gpu_layers_pref):
    if not os.path.exists(gguf_model_path):
        st.error(f"GGUF model file not found at: `{gguf_model_path}`.")
        st.stop()
    try:
        llm = Llama(
            model_path=gguf_model_path,
            n_gpu_layers=n_gpu_layers_pref, # Use the preference here
            n_ctx=8192,  # Increased context size for stability
            verbose=False,
            chat_format="llama-2"
        )
        return llm
    except Exception as e:
        st.error(f"A critical error occurred during model loading: {e}")
        raise

with st.spinner(f"Preparing AI model... this may take a moment on your system..."):
    try:
        model_llm = load_granite_model_cached(GGUF_MODEL_PATH, st.session_state.user_preferences["ai_n_gpu_layers"])
    except Exception as e:
        st.error(f"A critical error occurred during model loading: {e}")
        st.stop()

# --- Helper: Clean AI Output for Display ---
def clean_ai_output(text):
    """Removes common AI conversational prefixes from generated text."""
    if not isinstance(text, str):
        text = str(text)
    prefixes = ["[RESP]:", "AI Output:", "Response:", "Here is the output:", "Output:"]
    cleaned_text = text
    for prefix in prefixes:
        if cleaned_text.strip().startswith(prefix):
            cleaned_text = cleaned_text.strip()[len(prefix):].strip()
            break
    return cleaned_text

# --- AI Task Processing Functions (now return content instead of updating session_state directly) ---
def generate_sdlc_content(project_name, project_abstract, model_llm_obj, preferences):
    try:
        context_detail = f" with abstract:\n'{project_abstract}'" if project_abstract else ""
        sdlc_prompt = (
            f"Generate a concise Software Development Lifecycle (SDLC) outline for the project named '{project_name}'"
            f"{context_detail}. "
            "The outline should focus on the main phases: Requirements, Design, Development, Testing, and Deployment. "
            "For each phase, provide a short paragraph (2-3 sentences) describing relevant activities. "
            "Use clear bold headings for each phase (e.g., **Requirements**). "
            "Strictly adhere to this format. Do not include any conversational filler, "
            "introductions, conclusions, sub-headings, numbered lists, or bullet points. "
            "Start directly with the 'Requirements' heading."
        )
        context_message = {"role": "system", "content": "You are an AI assistant specialized in creating concise SDLC outlines."}
        messages = [context_message, {"role": "user", "content": sdlc_prompt}]
        output = model_llm_obj.create_chat_completion(
            messages=messages,
            max_tokens=int(preferences["ai_max_new_tokens"] * 1.5),
            temperature=preferences["ai_temperature"], # Use general AI temperature
            top_p=preferences["ai_top_p"],
            top_k=preferences["ai_top_k"],
            repeat_penalty=preferences["ai_repetition_penalty"],
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        return output["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Failed to generate SDLC: {e}")

def generate_project_abstract(project_name, user_description, model_llm_obj, preferences):
    try:
        description_part = f" It is further described as: {user_description}" if user_description.strip() else ""
        abstract_prompt = (
            f"You are an AI assistant skilled in writing clear and concise project abstracts. "
            f"Generate a professional and informative project abstract for a project titled '{project_name}'. "
            f"{description_part}. "
            f"The abstract should be 3-5 sentences long, summarizing the project's purpose, key features, and anticipated benefits or impact. "
            f"Do not include conversational filler or introductory phrases. Start directly with the abstract content."
        )
        messages = [{"role": "user", "content": abstract_prompt}]
        output = model_llm_obj.create_chat_completion(
            messages=messages,
            max_tokens=preferences["ai_max_new_tokens"],
            temperature=preferences["ai_temperature"],
            top_p=preferences["ai_top_p"],
            top_k=preferences["ai_top_k"],
            repeat_penalty=preferences["ai_repetition_penalty"],
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"Failed to generate abstract: {e}")

def perform_code_review(code_content, model_llm_obj, preferences):
    try:
        review_prompt = (
            f"You are an AI code review assistant, providing helpful and constructive feedback. "
            f"Review the following code snippet for potential bugs, style issues, "
            f"performance bottlenecks, security vulnerabilities, and adherence to best practices. "
            f"Provide concise, actionable feedback. If no issues are found, state that you found no critical issues. "
            f"Start directly with 'Code Review Results:' and use markdown for code examples if necessary. "
            f"Avoid conversational filler.\n\n"
            f"```\n{code_content}\n```\n\n"
            "Code Review Results:"
        )
        messages = [{"role": "user", "content": review_prompt}]
        output = model_llm_obj.create_chat_completion(
            messages=messages,
            max_tokens=preferences["ai_max_new_tokens"],
            temperature=preferences["ai_temperature"],
            top_p=preferences["ai_top_p"],
            top_k=preferences["ai_top_k"],
            repeat_penalty=preferences["ai_repetition_penalty"],
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"Failed to perform code review: {e}")

def generate_documentation(input_content, model_llm_obj, preferences):
    try:
        doc_prompt = (
            f"You are an AI assistant specialized in generating clear and concise technical documentation. "
            f"Your task is to create documentation for the following content. "
            f"If the content is code, provide docstrings (e.g., Python, Javadoc) or inline comments, along with a high-level explanation of its purpose, usage, parameters, and return values. "
            f"If the content is a description, elaborate on the concept, its purpose, and key aspects. "
            f"Format the documentation using Markdown, with appropriate headings (##), bullet points (-), and code blocks (```). "
            f"Start directly with the documentation content, do not include any conversational phrases like 'Here is the documentation:' or 'I have generated the documentation for you.'. "
            f"Be formal and technical.\n\n"
            f"Content to document:\n```\n{input_content}\n```\n\n"
            "Documentation:"
        )
        messages = [{"role": "user", "content": doc_prompt}]
        output = model_llm_obj.create_chat_completion(
            messages=messages,
            max_tokens=preferences["ai_max_new_tokens"] * 2,
            temperature=preferences["ai_temperature"],
            top_p=preferences["ai_top_p"],
            top_k=preferences["ai_top_k"],
            repeat_penalty=preferences["ai_repetition_penalty"],
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"Failed to generate documentation: {e}")

def analyze_and_resolve_bug(bug_description_or_code, model_llm_obj, preferences):
    try:
        bug_prompt = (
            f"You are an AI expert in identifying and resolving software bugs. "
            f"Analyze the following bug description, code snippet, or error message. "
            f"Your response should clearly state the likely cause of the bug, "
            f"explain the underlying issue, and provide a concrete, actionable solution or troubleshooting steps. "
            f"If a code fix is necessary, include the corrected code snippet in a markdown code block. "
            f"Focus on practical and direct solutions. "
            f"Start directly with 'Bug Analysis:' followed by the explanation and then 'Suggested Solution:'. "
            f"Do not include any conversational filler like 'I have analyzed the bug:'.\n\n"
            f"Bug details:\n```\n{bug_description_or_code}\n```\n\n"
            "Bug Analysis:"
        )
        messages = [{"role": "user", "content": bug_prompt}]
        output = model_llm_obj.create_chat_completion(
            messages=messages,
            max_tokens=preferences["ai_max_new_tokens"] * 2,
            temperature=preferences["ai_temperature"],
            top_p=preferences["ai_top_p"],
            top_k=preferences["ai_top_k"],
            repeat_penalty=preferences["ai_repetition_penalty"],
            stop=["<|eot_id|>", "<|end_of_text|>"]
        )
        return output["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise Exception(f"Failed to analyze bug: {e}")

# --- Main UI Rendering ---
st.title("SmartSDLC AI Assistant")
# make an AI hallucination disclaimer
st.write("**Disclaimer:** This AI assistant is designed to assist with the Software Development Lifecycle (SDLC) and related tasks. It is not a substitute for professional software development practices. Always review AI-generated content critically and ensure it meets your project's requirements and standards.")

# --- Project Management Section ---
st.header("Project Management")
with st.expander("Project Management", expanded=True):
    # Project creation and selection
    if "new_project_name_input_value" not in st.session_state:
        st.session_state.new_project_name_input_value = ""
    new_project_name_current_input = st.text_input(
        "Enter new project name:",
        value=st.session_state.new_project_name_input_value,
        key="new_project_name_sidebar_input",
        placeholder="New Project"
    )
    if st.button("Create New Project", key="create_new_project_btn"):
        if not new_project_name_current_input.strip():
            st.warning("Please enter a project name.")
        else:
            new_project_id = str(uuid.uuid4())
            st.session_state.projects[new_project_id] = get_default_project_data(
                project_name=new_project_name_current_input
            )
            st.session_state.current_project_id = new_project_id
            save_projects(st.session_state.projects)
            st.toast(f"Project '{new_project_name_current_input}' created!", icon="✅")
            st.session_state.new_project_name_input_value = ""
            st.rerun()
    st.markdown("---")
    # Project selection
    if st.session_state.projects:
        project_options = {proj_data["name"]: proj_id for proj_id, proj_data in st.session_state.projects.items()}
        project_names = list(project_options.keys())
        if st.session_state.current_project_id not in st.session_state.projects:
            if project_names:
                st.session_state.current_project_id = project_options[project_names[0]]
            else:
                st.session_state.current_project_id = None
        selected_project_name = None
        if st.session_state.current_project_id:
            try:
                current_project_name_for_select = st.session_state.projects[st.session_state.current_project_id]["name"]
                default_index = project_names.index(current_project_name_for_select)
            except (KeyError, ValueError):
                default_index = 0
            selected_project_name = st.selectbox(
                "Select an Existing Project:",
                options=project_names,
                index=default_index,
                key="project_selector",
                disabled=not bool(project_names)
            )
        else:
            st.selectbox(
                "Select an Existing Project:",
                options=["No Projects Available"],
                index=0,
                key="project_selector",
                disabled=True
            )
        if selected_project_name and st.session_state.current_project_id != project_options.get(selected_project_name):
            st.session_state.current_project_id = project_options.get(selected_project_name)
            st.rerun()
    if st.session_state.current_project_id:
        st.info(f"**Current Project:** {st.session_state.projects[st.session_state.current_project_id]['name']}")
        if st.button("Delete Selected Project", key="delete_project_btn"):
            if st.session_state.current_project_id:
                del st.session_state.projects[st.session_state.current_project_id]
                st.session_state.current_project_id = None
                save_projects(st.session_state.projects)
                st.toast("Project deleted!", icon="🗑️")
                st.rerun()

# --- AI Settings Section ---
with st.expander("AI Settings", expanded=False):
    st.session_state.user_preferences["ai_temperature"] = st.slider(
        "AI Temperature (Creativity)",
        min_value=0.0, max_value=1.0, value=st.session_state.user_preferences["ai_temperature"], step=0.01,
        key="ai_temp_slider", on_change=on_preference_change
    )
    st.session_state.user_preferences["ai_max_new_tokens"] = st.number_input(
        "Max Response Length (Tokens)",
        min_value=50, max_value=8192, value=st.session_state.user_preferences["ai_max_new_tokens"], step=50,
        key="ai_max_tokens_input", on_change=on_preference_change
    )
    st.session_state.user_preferences["ai_top_p"] = st.slider(
        "Top P (Nucleus Sampling)",
        min_value=0.0, max_value=1.0, value=st.session_state.user_preferences["ai_top_p"], step=0.01,
        key="ai_top_p_slider", on_change=on_preference_change
    )
    st.session_state.user_preferences["ai_top_k"] = st.number_input(
        "Top K",
        min_value=1, max_value=100, value=st.session_state.user_preferences["ai_top_k"], step=1,
        key="ai_top_k_input", on_change=on_preference_change
    )
    st.session_state.user_preferences["ai_repetition_penalty"] = st.slider(
        "Repetition Penalty",
        min_value=1.0, max_value=2.0, value=st.session_state.user_preferences["ai_repetition_penalty"], step=0.01,
        key="ai_rep_penalty_slider", on_change=on_preference_change
    )
    st.session_state.user_preferences["ai_n_gpu_layers"] = st.number_input(
        "GPU Layers (0 for CPU, -1 for all)",
        min_value=-1, max_value=999, value=st.session_state.user_preferences["ai_n_gpu_layers"], step=1,
        key="ai_n_gpu_layers_input", on_change=on_preference_change,
        help="Number of model layers to offload to GPU. Set to 0 for CPU-only. Use -1 to offload all layers if your GPU has enough VRAM. Requires `llama-cpp-python` to be compiled with GPU support."
    )
    st.markdown("""
        <small>Changes saved automatically.<br><i>Temperature:</i> Higher = more creative/random.<br><i>Top P/K:</i> Control diversity.<br><i>Repetition Penalty:</i> Prevents repetitive phrases.<br><i>GPU Layers:</i> Offload layers to GPU for faster inference (requires compatible GPU and `llama-cpp-python` installation).</small>""", unsafe_allow_html=True)

# --- Main Content Area Logic ---
if st.session_state.current_project_id is None:
    st.info("Create a new project or select an existing one to begin.")
else:
    current_project = st.session_state.projects[st.session_state.current_project_id]
    st.header(f"Project: {current_project['name']}")
    # Hide tools if a task was just submitted or a task is running
    if st.session_state.get("ai_task_just_submitted", False) or st.session_state.get("ai_running_task"):
        st.warning("AI is generating the response. It may take a while, wait until the task is completed before using the tools.")
        # Reset the flag after showing the warning
        st.session_state.ai_task_just_submitted = False
    else:
        st.markdown("### Project Tools")

        # --- SDLC Generation & Display ---
        with st.expander("🛠️ SDLC Phases", expanded=False):
            sdlc_button_key_main = f"generate_sdlc_button_{st.session_state.current_project_id}_main"
            regenerate_sdlc_button_key_main = f"regenerate_sdlc_button_{st.session_state.current_project_id}_main"
            # Display current SDLC content if available
            if current_project.get("sdlc_phases_content"):
                st.markdown(clean_ai_output(current_project["sdlc_phases_content"])) # Clean output for display
                if st.button("Regenerate SDLC", key=regenerate_sdlc_button_key_main):
                    st.toast("Regenerating SDLC outline...", icon="🛠️")
                    # Clear existing content and add task
                    current_project["sdlc_phases_content"] = None
                    st.session_state.projects[st.session_state.current_project_id] = current_project
                    save_projects(st.session_state.projects)
                    add_ai_task(
                        'sdlc',
                        current_project['abstract'],
                        current_project['name'],
                        tool_label="SDLC Phases"
                    )
            else:
                if st.button("Generate SDLC Outline", key=sdlc_button_key_main):
                    st.toast("Generating SDLC outline...", icon="🛠️")
                    add_ai_task(
                        'sdlc',
                        current_project['abstract'],
                        current_project['name'],
                        tool_label="SDLC Phases"
                    )
                # Display pending/running status from the queue directly if the task is there
                pending_sdlc_task = next((task for task in st.session_state.ai_pending_tasks if task['type'] == 'sdlc' and task['project_name'] == current_project['name']), None)
                running_sdlc_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'sdlc' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

                if running_sdlc_task and running_sdlc_task['status'] == 'running':
                    st.info("SDLC is currently being generated...")
                elif pending_sdlc_task:
                    st.info("SDLC generation is pending...")
                elif (running_sdlc_task and running_sdlc_task['status'] == 'failed'):
                    st.error(f"Failed to generate SDLC: {clean_ai_output(running_sdlc_task['error'])}")
                elif not pending_sdlc_task and not running_sdlc_task: # Only show 'not generated yet' if no task is pending/running
                    st.info("SDLC phases have not been generated for this project yet.")


        # --- Abstract Generation Section ---
        with st.expander("✨ Abstract Generator", expanded=False):
            st.write("Generate a professional project abstract for your project.")
            # Make keys unique per section and per widget type
            abstract_gen_project_name_input_key_tools = f"abstract_gen_project_name_input_widget_{st.session_state.current_project_id}_tools"
            abstract_gen_description_input_key_tools = f"abstract_gen_description_input_widget_{st.session_state.current_project_id}_tools"
            generate_abstract_btn_key_tools = f"generate_abstract_btn_{st.session_state.current_project_id}_tools"

            st.session_state.abstract_gen_project_name_input = st.text_input(
                "Project Name for Abstract (Optional, defaults to current project name):",
                value=current_project['name'],
                key=abstract_gen_project_name_input_key_tools
            )
            st.session_state.abstract_gen_description_input = st.text_area(
                "Brief Description or Keywords (Optional):",
                value=st.session_state.abstract_gen_description_input,
                height=100,
                key=abstract_gen_description_input_key_tools
            )
            if st.button("Generate Abstract", key=generate_abstract_btn_key_tools):
                st.toast("Generating project abstract...", icon="✨")
                add_ai_task(
                    'abstract',
                    st.session_state.abstract_gen_description_input,
                    st.session_state.abstract_gen_project_name_input,
                    tool_label="Abstract Generator"
                )
                # Clear input fields after adding task
                st.session_state.abstract_gen_description_input = ""
                st.session_state.abstract_gen_project_name_input = current_project['name'] # Reset to default
                st.session_state.abstract_gen_result = "" # Clear previous result
                st.session_state.abstract_gen_status = "not_started" # Reset status for new generation

            # Check for abstract generation task in queue
            pending_abstract_task = next(
                (task for task in reversed(st.session_state.ai_pending_tasks)
                 if task['type'] == 'abstract' and task['project_name'] == current_project['name']),
                None
            )
            running_abstract_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'abstract' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

            # Find the most recent 'abstract' task for the current project in history
            recent_abstract_history_task = next(
                (task for task in reversed(st.session_state.ai_finished_tasks)
                 if task['type'] == 'abstract' and task['project_name'] == current_project['name'] and task['status'] == 'done'),
                None
            )

            # Determine the task to display (running > pending > history)
            task_to_display = running_abstract_task or pending_abstract_task or recent_abstract_history_task

            if running_abstract_task and running_abstract_task['status'] == 'running':
                st.info("AI is generating the abstract...")
            elif pending_abstract_task:
                st.info("Abstract generation is pending...")
            elif task_to_display and task_to_display['status'] == 'failed':
                st.error(clean_ai_output(task_to_display['error'])) # Clean output for display


        # --- Code Review Section ---
        with st.expander("📝 Code Review Assistant", expanded=False):
            st.write("This section provides AI-powered code review suggestions.")
            code_input_review_area_key_tools = f"code_input_review_area_{st.session_state.current_project_id}_tools"
            perform_code_review_btn_key_tools = f"perform_code_review_btn_{st.session_state.current_project_id}_tools"

            st.session_state.code_input_review = st.text_area(
                "Paste your code here for review:",
                value=st.session_state.code_input_review,
                height=300,
                key=code_input_review_area_key_tools
            )
            if st.button("Perform Code Review", key=perform_code_review_btn_key_tools):
                st.toast("Performing code review...", icon="📝")
                add_ai_task(
                    'code_review',
                    st.session_state.code_input_review,
                    current_project['name'],
                    tool_label="Code Review"
                )
                st.session_state.code_input_review = "" # Clear input after submission
                st.session_state.code_review_result = "" # Clear previous result
                st.session_state.code_review_status = "not_started" # Reset status for new generation

            pending_code_review_task = next(
                (task for task in reversed(st.session_state.ai_pending_tasks)
                 if task['type'] == 'code_review' and task['project_name'] == current_project['name']),
                None
            )
            running_code_review_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'code_review' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

            recent_code_review_history_task = next(
                (task for task in reversed(st.session_state.ai_finished_tasks)
                     if task['type'] == 'code_review' and task['project_name'] == current_project['name'] and task['status'] == 'done'),
                None
            )

            task_to_display = running_code_review_task or pending_code_review_task or recent_code_review_history_task

            if running_code_review_task and running_code_review_task['status'] == 'running':
                st.info("AI is reviewing the code...")
            elif pending_code_review_task:
                st.info("Code review is pending...")
            elif task_to_display and task_to_display['status'] == 'done':
                st.subheader("AI Code Review Results:")
                st.markdown(clean_ai_output(task_to_display['result'])) # Clean output for display
                clear_code_review_btn_key_tools = f"clear_code_review_btn_{task_to_display['id']}_tools"
                if st.button("Clear Review Results", key=clear_code_review_btn_key_tools):
                    st.toast("Clearing code review results...", icon="🧹")
                    st.session_state.code_review_result = "" # Clear UI display
                    st.session_state.code_review_status = "not_started"
                    st.rerun() # Rerun to reflect clearance
            elif task_to_display and task_to_display['status'] == 'failed':
                st.error(clean_ai_output(task_to_display['error'])) # Clean output for display


        # --- Documentation Assistant Section ---
        with st.expander("📄 Documentation Assistant", expanded=False):
            st.write("This section will help generate and update project documentation automatically.")
            doc_input_area_key_tools = f"doc_input_area_{st.session_state.current_project_id}_tools"
            generate_doc_btn_key_tools = f"generate_doc_btn_{st.session_state.current_project_id}_tools"

            st.session_state.doc_input = st.text_area(
                "Paste code or describe what you need documented:",
                value=st.session_state.doc_input,
                height=250,
                key=doc_input_area_key_tools
            )
            if st.button("Generate Documentation", key=generate_doc_btn_key_tools):
                st.toast("Generating documentation...", icon="📄")
                add_ai_task(
                    'documentation',
                    st.session_state.doc_input,
                    current_project['name'],
                    tool_label="Documentation"
                )
                st.session_state.doc_input = "" # Clear input
                st.session_state.doc_result = "" # Clear previous result
                st.session_state.doc_status = "not_started" # Reset status

            pending_doc_task = next(
                (task for task in reversed(st.session_state.ai_pending_tasks)
                 if task['type'] == 'documentation' and task['project_name'] == current_project['name']),
                None
            )
            running_doc_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'documentation' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

            recent_doc_history_task = next(
                (task for task in reversed(st.session_state.ai_finished_tasks)
                     if task['type'] == 'documentation' and task['project_name'] == current_project['name'] and task['status'] == 'done'),
                None
            )

            task_to_display = running_doc_task or pending_doc_task or recent_doc_history_task

            if running_doc_task and running_doc_task['status'] == 'running':
                st.info("AI is generating documentation...")
            elif pending_doc_task:
                st.info("Documentation generation is pending...")
            elif task_to_display and task_to_display['status'] == 'done':
                st.subheader("AI Generated Documentation:")
                st.markdown(clean_ai_output(task_to_display['result'])) # Clean output for display
                clear_doc_btn_key_tools = f"clear_doc_btn_{task_to_display['id']}_tools"
                if st.button("Clear Documentation Results", key=clear_doc_btn_key_tools):
                    st.toast("Clearing documentation results...", icon="🧹")
                    st.session_state.doc_result = "" # Clear UI display
                    st.session_state.doc_status = "not_started"
                    st.rerun()
            elif task_to_display and task_to_display['status'] == 'failed':
                st.error(clean_ai_output(task_to_display['error'])) # Clean output for display


        # --- Bug Detection & Resolution Section ---
        with st.expander("🐞 Bug Detection & Resolution", expanded=False):
            st.write("This section will help identify potential bugs and suggest solutions.")
            bug_input_area_key_tools = f"bug_input_area_{st.session_state.current_project_id}_tools"
            analyze_bug_btn_key_tools = f"analyze_bug_btn_{st.session_state.current_project_id}_tools"

            st.session_state.bug_input = st.text_area(
                "Paste code, error messages, or describe the bug observed:",
                value=st.session_state.bug_input,
                height=250,
                key=bug_input_area_key_tools
            )
            if st.button("Analyze Bug & Suggest Solution", key=analyze_bug_btn_key_tools):
                st.toast("Analyzing bug and suggesting solution...", icon="🐞")
                add_ai_task(
                    'bug',
                    st.session_state.bug_input,
                    current_project['name'],
                    tool_label="Bug Detection"
                )
                st.session_state.bug_input = "" # Clear input
                st.session_state.bug_result = "" # Clear previous result
                st.session_state.bug_status = "not_started" # Reset status

            pending_bug_task = next(
                (task for task in reversed(st.session_state.ai_pending_tasks)
                 if task['type'] == 'bug' and task['project_name'] == current_project['name']),
                None
            )
            running_bug_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'bug' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

            recent_bug_history_task = next(
                (task for task in reversed(st.session_state.ai_finished_tasks)
                     if task['type'] == 'bug' and task['project_name'] == current_project['name'] and task['status'] == 'done'),
                None
            )

            task_to_display = running_bug_task or pending_bug_task or recent_bug_history_task

            if running_bug_task and running_bug_task['status'] == 'running':
                st.info("AI is analyzing the bug...")
            elif pending_bug_task:
                st.info("Bug analysis is pending...")
            elif task_to_display and task_to_display['status'] == 'done':
                st.subheader("AI Bug Analysis & Solution:")
                st.markdown(clean_ai_output(task_to_display['result'])) # Clean output for display
                clear_bug_btn_key_tools = f"clear_bug_btn_{task_to_display['id']}_tools"
                if st.button("Clear Bug Results", key=clear_bug_btn_key_tools):
                    st.toast("Clearing bug analysis results...", icon="🧹")
                    st.session_state.bug_result = "" # Clear UI display
                    st.session_state.bug_status = "not_started"
                    st.rerun()
            elif task_to_display and task_to_display['status'] == 'failed':
                st.error(clean_ai_output(task_to_display['error'])) # Clean output for display

        # --- Chat with AI Tool ---
        with st.expander("💬 Chat with AI", expanded=False):
            st.write("Ask any question related to your project and get an AI-powered answer.")
            chat_input_key = f"chat_with_ai_input_{st.session_state.current_project_id}"
            user_prompt = st.text_area(
                "Type your question for the AI:",
                value="",
                key=chat_input_key,
                height=100
            )
            if st.button("Ask AI", key=f"ask_ai_btn_{st.session_state.current_project_id}_tools") and user_prompt.strip():
                st.toast("Sending your question to AI...", icon="💬")
                add_ai_task(
                    'chat',
                    user_prompt,
                    current_project['name'],
                    tool_label="Chat with AI"
                )
                # Clear input after adding task
                st.session_state[chat_input_key] = ""
                st.rerun() # Rerun to clear input and potentially show "AI processing"

            # Display chat history for the current project
            st.subheader("Chat History:")
            # Find pending/running chat task for this project
            pending_chat_task = next(
                (task for task in reversed(st.session_state.ai_pending_tasks)
                 if task['type'] == 'chat' and task['project_name'] == current_project['name'] and task['status'] in ['pending']),
                None
            )
            running_chat_task = st.session_state.ai_running_task if st.session_state.ai_running_task and st.session_state.ai_running_task['type'] == 'chat' and st.session_state.ai_running_task['project_name'] == current_project['name'] else None

            if running_chat_task:
                st.info("AI is processing your chat request...")
            elif pending_chat_task:
                st.info("Chat request is pending...")

            if current_project["chat_history"]:
                # Display chat history from project data, newest first
                for chat_entry in reversed(current_project["chat_history"]):
                    st.markdown(f"**You:** {chat_entry['user']}")
                    st.markdown(f"**AI:** {clean_ai_output(chat_entry['ai'])}") # Clean output for display
                    st.markdown("---")
            else:
                st.info("No chat history yet.")


    # --- AI Task History (Finished Tasks) ---
    st.markdown("### AI Task History (Finished Tasks)")
    # --- Full-width output display for selected history item ---
    if st.session_state.display_task_output_id:
        st.markdown("---") # Separator for clarity
        with st.expander(st.session_state.display_task_output_title, expanded=True):
            st.markdown(st.session_state.display_task_output_content)
            if st.button("Hide Output", key="hide_full_output_btn"):
                st.session_state.display_task_output_id = None
                st.session_state.display_task_output_content = ""
                st.session_state.display_task_output_title = ""
                st.rerun()
    history = st.session_state.ai_finished_tasks
    if history:
        for idx, task in enumerate(history):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            with col1:
                st.write(f"**{task.get('tool_label', task['type'].capitalize())}**")
                st.caption(f"Project: {task.get('project_name', '')} ({task['status'].capitalize()})") # Show final status
                st.code(str(task.get('input', ''))[:120] + ("..." if len(str(task.get('input', ''))) > 120 else ""))
            with col2:
                st.write(f"Status: {task['status']}")
            with col3:
                if task['status'] == 'done' and task.get('result'):
                    if st.button("View Output", key=f"view_hist_output_{task['id']}"):
                        st.session_state.display_task_output_id = task['id']
                        st.session_state.display_task_output_content = clean_ai_output(task['result'])
                        st.session_state.display_task_output_title = f"Output for Task: {task.get('tool_label', task['type'].capitalize())} ({task['id']})"
                        st.rerun()
                elif task['status'] == 'failed' and task.get('error'):
                    if st.button("View Error", key=f"view_hist_error_{task['id']}"):
                        st.session_state.display_task_output_id = task['id']
                        st.session_state.display_task_output_content = f"Error: {clean_ai_output(task['error'])}"
                        st.session_state.display_task_output_title = f"Error for Task: {task.get('tool_label', task['type'].capitalize())} ({task['id']})"
                        st.rerun()
                elif task['status'] == 'cancelled':
                    st.info("Cancelled")
            with col4:
                if st.button("Delete", key=f"delete_hist_task_{task['id']}"): # Use task ID for unique key
                    st.session_state.ai_finished_tasks.pop(idx)
                    save_ai_finished_tasks(st.session_state.ai_finished_tasks)
                    st.toast("Task removed from history.", icon="🗑️")
                    st.rerun()
    else:
        st.info("No tasks in the AI history.")


    # --- AI Task Queue Processing (Refactored to new files logic) ---
    def process_ai_queue():
        st.info("🚀 Queue processor triggered")

        running_task = st.session_state.ai_running_task
        pending_tasks = st.session_state.ai_pending_tasks
        finished_tasks = st.session_state.ai_finished_tasks

        # Show queue state for debugging
        with st.expander("🔍 Queue Debug Info", expanded=False):
            st.subheader("Queue State Snapshot")
            st.write("Running Task:")
            st.json(running_task)
            st.write("Pending Tasks:")
            st.json(pending_tasks)
            st.write("Recently Finished Tasks:")
            st.json(finished_tasks[:1])

        # If a task is already running, exit
        if running_task is not None:
            st.info("⏳ A task is already running. Waiting for it to finish.")
            return
        st.info("🔄 Checking for pending tasks...")

        # If there's a pending task, promote and process it
        if pending_tasks:
            st.info("🎯 Promoting next pending task to running.")
            running_task = pending_tasks.pop(0)
            running_task['status'] = 'running'

            st.session_state.ai_running_task = running_task
            st.session_state.ai_pending_tasks = pending_tasks
            save_ai_pending_tasks(pending_tasks)
            save_ai_running_task(running_task)

            try:
                prefs = get_task_ai_settings(running_task['type'], st.session_state.user_preferences)
                result_content = None
                st.info(f"⚙️ Processing task of type: `{running_task['type']}`")

                if running_task['type'] == 'abstract':
                    result_content = generate_project_abstract(running_task['project_name'], running_task['input'], model_llm, prefs)
                elif running_task['type'] == 'code_review':
                    result_content = perform_code_review(running_task['input'], model_llm, prefs)
                elif running_task['type'] == 'documentation':
                    result_content = generate_documentation(running_task['input'], model_llm, prefs)
                elif running_task['type'] == 'bug':
                    result_content = analyze_and_resolve_bug(running_task['input'], model_llm, prefs)
                elif running_task['type'] == 'sdlc':
                    result_content = generate_sdlc_content(running_task['project_name'], running_task['input'], model_llm, prefs)
                    target_project_id = next((pid for pid, pdata in st.session_state.projects.items() if pdata['name'] == running_task['project_name']), None)
                    if target_project_id:
                        st.session_state.projects[target_project_id]["sdlc_phases_content"] = result_content
                        save_projects(st.session_state.projects)
                elif running_task['type'] == 'chat':
                    context_message = {
                        "role": "system",
                        "content": f"You are an expert software development assistant. Answer user questions directly and concisely, focusing on the project '{running_task['project_name']}'."
                    }
                    messages = [context_message, {"role": "user", "content": running_task['input']}]
                    output = model_llm.create_chat_completion(
                        messages=messages,
                        max_tokens=prefs["ai_max_new_tokens"],
                        temperature=prefs["ai_temperature"],
                        top_p=prefs["ai_top_p"],
                        top_k=prefs["ai_top_k"],
                        repeat_penalty=prefs["ai_repetition_penalty"],
                        stop=["<|eot_id|>", "<|end_of_text|>"]
                    )
                    result_content = output["choices"][0]["message"]["content"]
                    target_project_id = next((pid for pid, pdata in st.session_state.projects.items() if pdata['name'] == running_task['project_name']), None)
                    if target_project_id:
                        st.session_state.projects[target_project_id]['chat_history'].insert(0, {"user": running_task['input'], "ai": result_content})
                        save_projects(st.session_state.projects)

                running_task['result'] = result_content
                running_task['status'] = 'done'
                st.success(f"✅ Task `{running_task['type']}` completed successfully.")

            except Exception as e:
                running_task['status'] = 'failed'
                running_task['error'] = str(e)
                st.error(f"❌ Task `{running_task['type']}` failed: {e}")

            finally:
                finished_tasks.insert(0, running_task)
                st.session_state.ai_finished_tasks = finished_tasks
                st.session_state.ai_running_task = None
                save_ai_running_task(None)
                save_ai_finished_tasks(finished_tasks)
                st.rerun()  # ✅ Must be the last thing that always happens


    # --- One-Time Per-Render Flag to Avoid Double-Processing ---
    # Only process if a task is running, or if there are pending tasks and nothing is running
    if not st.session_state.get("queue_processed_this_run", False):
        if st.session_state.ai_running_task is not None or (st.session_state.ai_pending_tasks and st.session_state.ai_running_task is None):
            st.session_state.queue_processed_this_run = True
            process_ai_queue()
