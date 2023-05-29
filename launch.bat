@echo off
cd %USERPROFILE%\Desktop\assistant
call .\.venv\Scripts\activate.bat
python speech_assistant.py
pause