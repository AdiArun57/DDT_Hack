# Setup Virtual Environment for CreditBridge
Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Environment setup complete! Use '.\venv\Scripts\Activate.ps1' to activate it in the future." -ForegroundColor Green
