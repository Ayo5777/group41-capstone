param(
  [Parameter(Mandatory=$true)]
  [string]$Server,

  [int]$NumClients = 10,
  [switch]$Cpu,
  [int]$BatchSize = 32,
  [int]$Seed = 42,
  [string]$DataDir = "DataPart\data",

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
Write-Host "BatchSize=$BatchSize Seed=$Seed DataDir=$DataDir Cpu=$Cpu"
Write-Host "Logs: logs\client_<id>.out.log and logs\client_<id>.err.log"
Write-Host "Stop: run scripts\kill_clients.ps1"

for ($i = 0; $i -lt $NumClients; $i++) {
  $stdoutLog = "logs\client_$i.out.log"
  $stderrLog = "logs\client_$i.err.log"

  $argsList = @(
    $clientScript,
    "--client_id", "$i",
    "--num_clients", "$NumClients",
    "--server", "$Server",
    "--batch_size", "$BatchSize",
    "--seed", "$Seed",
    "--data_dir", "$DataDir"
  )

  if ($Cpu) {
    $argsList += "--cpu"
  }

  Start-Process -NoNewWindow -FilePath "python" -ArgumentList $argsList `
    -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
  Start-Sleep -Milliseconds 150

  Write-Host "Client $i started"
}

Write-Host "Done."
