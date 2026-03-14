@echo off
setlocal

:: Check for Git installation
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed or not in your PATH.
    echo Please install Git from https://git-scm.com/downloads/
    echo After installing, restart your terminal and run this script again.
    pause
    exit /b
)

echo [INFO] Initializing Git repository...
if not exist .git (
    git init
    git branch -M main
) else (
    echo [INFO] Git repository already initialized.
)

:: Configure user if not set (optional prompts)
git config user.name >nul 2>&1
if %errorlevel% neq 0 (
    set /p GIT_USER="Enter your Git username: "
    set /p GIT_EMAIL="Enter your Git email: "
    git config user.name "%GIT_USER%"
    git config user.email "%GIT_EMAIL%"
)

echo [INFO] Adding files...
git add .

echo [INFO] Committing changes...
git commit -m "Initial commit: Optimized multi-agent robot coordination system"

echo [INFO] Setting up remote repository...
git remote remove origin >nul 2>&1
git remote add origin https://github.com/DAIJINGFU/ROBOT.git

echo [INFO] Pushing code to GitHub...
git push -u origin main

if %errorlevel% neq 0 (
    echo [ERROR] Failed to push to GitHub. Check your credentials or internet connection.
    echo Ensure you are logged in or using a Personal Access Token.
) else (
    echo [SUCCESS] Code pushed successfully!
)

pause
