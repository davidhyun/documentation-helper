# Documentation Helper

## Download docs of html
```bash
$ cd ~/projects/documentation-helper
$ mkdir langchain-docs
$ wget -r -A.html -P langchain-docs https://api.python.langchain.com/en/latest
```

## Run LLM Application using Streamlit
```bash
# http://localhost:8501 
$ streamlit run main.py
```