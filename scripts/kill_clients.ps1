# Kills python processes that are running client scripts
# Safe blunt tool for demos
Get-Process python -ErrorAction SilentlyContinue | Where-Object {
  $_.Path -ne $null
} | ForEach-Object { }

# More reliable: kill by commandline containing "client.py"
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
  $_.CommandLine -match "fl_clients\\.*client\.py"
} | ForEach-Object {
  Write-Host "Killing PID $($_.ProcessId): $($_.CommandLine)"
  Stop-Process -Id $_.ProcessId -Force
}
