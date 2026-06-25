Write-Host "Starting python server..."
Start-Process -FilePath "python.exe" -ArgumentList "Game/game.py" -NoNewWindow

Write-Host "Starting html page"
Start-Process "Game/index.html" -PassThru