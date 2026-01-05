@echo off
REM ============================================================================
REM Fashion Image Similarity - MongoDB Image Upload Batch Script
REM ============================================================================
REM This script automatically uploads all images from the fashion-dataset folder
REM to MongoDB Atlas and stores metadata for the web application.
REM
REM Prerequisites:
REM   - Python 3.9+ installed and in PATH
REM   - models/upload_images_from_disk.py in same directory
REM   - .env file with MONGO_URI in project root
REM   - fashion-dataset folder containing images
REM ============================================================================

echo.
echo ============================================================================
echo Fashion Image Similarity - MongoDB Upload Assistant
echo ============================================================================
echo.

REM Change to script directory (upload folder)
cd /d "%~dp0"
if errorlevel 1 (
    echo Error: Could not navigate to upload directory.
    pause
    exit /b 1
)

echo Working directory: %cd%
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.9+ from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM Check if .env file exists
if not exist ".env" (
    echo Error: .env file not found in %cd%
    echo Please ensure .env file contains MONGO_URI and other configuration.
    pause
    exit /b 1
)
echo .env file found: OK
echo.

REM Check if upload script exists
if not exist "upload_images_from_disk.py" (
    echo Error: upload_images_from_disk.py not found.
    pause
    exit /b 1
)
echo Upload script found: OK
echo.

REM Get dataset path from user
echo.
echo Enter the path to your fashion-dataset folder:
echo Example: D:\datasets\fashion-dataset
echo.
set /p DATASET_PATH="Dataset path: "

REM Verify dataset path exists
if not exist "%DATASET_PATH%" (
    echo Error: Dataset path does not exist: %DATASET_PATH%
    pause
    exit /b 1
)

if not exist "%DATASET_PATH%\train" (
    echo Error: %DATASET_PATH%\train folder not found.
    echo Ensure the fashion-dataset has a 'train' subfolder.
    pause
    exit /b 1
)

echo Dataset path verified: %DATASET_PATH%
echo.

REM Install required packages
echo ============================================================================
echo Installing required Python packages...
echo ============================================================================
pip install pymongo python-dotenv --quiet
if errorlevel 1 (
    echo Warning: Some packages may not have installed correctly.
    echo Attempting to continue anyway...
)
echo.

REM Run the upload script
echo ============================================================================
echo Starting image upload to MongoDB Atlas...
echo ============================================================================
echo This may take 2-5 hours depending on your internet connection.
echo Do NOT close this window until the upload completes.
echo.
echo Progress will be shown below:
echo ============================================================================
echo.

python upload_images_from_disk.py "%DATASET_PATH%"

if errorlevel 1 (
    echo.
    echo ============================================================================
    echo Error: Upload script failed. Check the error messages above.
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo SUCCESS: Image upload completed!
echo ============================================================================
echo.
echo Next steps:
echo   1. Copy .env and gallery.json back to your main project
echo   2. Run: uvicorn backend.main:app --reload
echo   3. Open http://localhost:8000 in your browser
echo   4. Test the similarity search with your uploaded images
echo.
echo MongoDB Collections Updated:
echo   - fashion_db.images (with binary image_data fields)
echo.
pause
exit /b 0
