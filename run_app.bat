@echo off
call .venv_clean\Scripts\activate
python -m streamlit run app.py
pause
