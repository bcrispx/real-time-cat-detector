@echo off
:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the classifier
python realtime_classifier.py

:: Keep the window open if there's an error
if errorlevel 1 pause

:: Deactivate virtual environment
deactivate
