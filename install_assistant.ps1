# Download and install Chocolatey (a package manager for Windows)
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))

# Install Git, MPV, and Python using Chocolatey
choco install git mpv python --version 3.11.3 -y

# Clone the GitHub repository
$repoUrl = "https://github.com/rheaton64/VoiceAssistantGUI"
$destination = "$env:USERPROFILE\Desktop\assistant"
git clone $repoUrl $destination

# Create a Python virtual environment
Set-Location $destination
python -m venv venv

# Activate the virtual environment
. .\venv\Scripts\Activate

# Install packages from requirements.txt
pip install -r requirements.txt