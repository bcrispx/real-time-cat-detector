@echo off
echo Building Cat Detector executable...
pyinstaller --clean cat_detector.spec
echo Done!
pause
