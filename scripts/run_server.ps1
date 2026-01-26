param(
  [string]$Host = "0.0.0.0",
  [int]$Port = 8080,
  [int]$Rounds = 3
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$preferred = @("server\server.py", "server\new_server.py", "server\simple_server.py")
$serverScript = $null
foreach ($s in $preferred) {
  if (Test-Path $s) { $serverScript = $s; break }
}

if ($null -eq $serverScript) {
  Write-Host "ERROR: Could not find any server script: $($preferred -join ', ')"
  exit 1
}

Write-Host "Starting server: $serverScript"
Write-Host "Host=$Host Port=$Port Rounds=$Rounds"

# Pass common args; adjust later if script differs
python $serverScript --host $Host --port $Port --rounds $Rounds
