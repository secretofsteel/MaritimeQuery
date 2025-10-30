# Maritime RAG Streamlit App

## Local Setup

### Option A – Python `venv`

1. **Create a virtual environment**
   - PowerShell (recommended on Windows):
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - Git Bash / WSL:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```
2. **Install dependencies**
   ```powershell
   python -m pip install "pip<24.1"  # Workaround for textract packaging bug
   pip install -r requirements.txt
   ```
3. **Configure environment**
   - Set your Gemini API key:
     - PowerShell:
       ```powershell
       setx GOOGLE_API_KEY "your-google-api-key"
       ```
       Restart the terminal (or run `$env:GOOGLE_API_KEY="your-google-api-key"` for the current session).
     - Git Bash / WSL:
       ```bash
       export GOOGLE_API_KEY="your-google-api-key"
       ```
   - Optional: set custom paths
     - PowerShell:
       ```powershell
       setx MARITIME_RAG_DOCS "C:\path\to\docs"
       setx MARITIME_RAG_CHROMA "C:\path\to\chroma"
       setx MARITIME_RAG_CACHE "C:\path\to\cache"
       ```
     - Git Bash / WSL:
       ```bash
       export MARITIME_RAG_DOCS="/path/to/docs"
       export MARITIME_RAG_CHROMA="/path/to/chroma"
       export MARITIME_RAG_CACHE="/path/to/cache"
       ```
4. **Prepare documents**
   - Place maritime documents under `data/docs` or the folder defined by `MARITIME_RAG_DOCS`.

5. **Run the app**
   - Classic single-file version (unchanged fallback):
     ```powershell
     streamlit run streamlit_app.py
     ```
   - Modular refactor (recommended):
     ```powershell
     streamlit run rag_modular.py
     ```
   - Read-only tester (no library management):
      ```powershell
      streamlit run streamlit_app_viewer.py
      ```

### Option B – Miniconda / Anaconda

1. **Create and activate environment**
   ```powershell
   conda create -n maritime-rag python=3.10
   conda activate maritime-rag
   ```
2. **Install dependencies**
   ```powershell
   python -m pip install "pip<24.1"
   pip install -r requirements.txt
   ```
3. **Configure environment variables**
   ```powershell
   conda env config vars set GOOGLE_API_KEY="your-gemini-key"
   conda env config vars set COHERE_API_KEY="your-cohere-key"  # optional
   conda deactivate
   conda activate maritime-rag
   ```
4. **Prepare documents & run**
   - Place documents under `data/docs` (or custom path) and launch with one of the entrypoints:
     ```powershell
     streamlit run rag_modular.py
     # or
     streamlit run streamlit_app.py
     # or (read-only mode)
     streamlit run streamlit_app_viewer.py
     ```

### Option C – Docker (no prior experience)

1. **Install Docker Desktop**
   - Download from https://www.docker.com/products/docker-desktop/ and install (accept the defaults).  
   - After installation, launch Docker Desktop once so it keeps running in the background.

2. **Open a terminal in this project folder**
   - Windows: open PowerShell and run:
     ```powershell
     cd C:\path\to\MaritimeRAG
     ```

3. **Put your documents in a local folder**
   - Example: create `C:\MaritimeData\docs` and drop all PDFs/Word/Excel files there.

4. **Create an `.env` file for secrets (in the project root)**
   ```bash
   # make sure the file is named exactly ".env"
   GOOGLE_API_KEY=your-google-api-key
   # optional paths inside the container (leave defaults unless you know why to change them)
   # MARITIME_RAG_DOCS=/data/docs
   # MARITIME_RAG_CACHE=/data/cache
   # MARITIME_RAG_CHROMA=/data/chroma_db
   ```

5. **Build the Docker image (one-time)**
   ```powershell
   docker build -t maritime-rag .
   ```

6. **Run the container**
   ```powershell
   docker run ^
     --name maritime-rag-app ^
     --env-file .env ^
     -p 8501:8501 ^
     -v C:\MaritimeData\docs:/data/docs ^
     -v C:\MaritimeData\cache:/data/cache ^
     -v C:\MaritimeData\chroma:/data/chroma_db ^
     maritime-rag
   ```
   - The `-v` flags link your local folders into the container. Docker will create the folders if they do not already exist.
   - Keep the terminal open; the logs will show Streamlit starting.

7. **Open the app**
   - Navigate to http://localhost:8501 in your browser.

8. **Stop the app**
   - Press `Ctrl+C` in the terminal or run:
     ```powershell
     docker stop maritime-rag-app
     docker rm maritime-rag-app
     ```

## Streamlit Cloud (Beta)

1. Fork this repository.
2. In Streamlit Cloud, create a new app and point it to `streamlit_app.py`.
3. Set the `GOOGLE_API_KEY` (and optional overrides) in the app’s Secrets manager.
4. Upload or mount your document library into the configured docs path.
