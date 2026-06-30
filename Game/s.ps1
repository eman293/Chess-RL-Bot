# C:\Users\eman2\Documents\GitHub\Project\.venv\Scripts\Activate.ps1

Write-Host "Starting python server..."
Start-Process -FilePath "python.exe" -ArgumentList "game.py" -NoNewWindow

Write-Host "Starting html page"
Start-Process "index.html" -PassThru