param(
  [Parameter(Mandatory=$true)]
  [string]$Server,

  [int]$NumClients = 15,

  [string]$ClientScriptPreferred = "fl_clients\new_client.py",
  [string]$ClientScriptFallback  = "fl_clients\simple_client.py"
)

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

# Pick client script
$clientScript = $ClientScriptPreferred
if (-not (Test-Path $clientScript)) {
  $clientScript = $ClientScriptFallback
}
if (-not (Test-Path $clientScript)) {
  Write-Host "ERROR: No client script found at $ClientScriptPreferred or $ClientScriptFallback"
  exit 1
}

Write-Host "Starting $NumClients clients using $clientScript -> $Server"
Write-Host "Logs: logs\client_<id>.log"
Write-Host "Stop: run scripts\kill_clients.ps1"

for ($i = 0; $i -lt $NumClients; $i++) {
  $log = "logs\client_$i.log"

  # We don't know exact CLI args each teammate used, so we pass common ones.
  # If your client script doesn't accept these, you'll adjust later.
  $argsList = @($clientScript, "--client_id", "$i", "--num_clients", "$NumClients", "--server", "$Server")

  Start-Process -NoNewWindow -FilePath "python" -ArgumentList $argsList `
    -RedirectStandardOutput $log -RedirectStandardError $log
  Start-Sleep -Milliseconds 150
}

Write-Host "Done."
