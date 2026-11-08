param([switch]$NoBrowser)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath = (Get-Command python -ErrorAction Stop).Source
$serverArgs = @((Join-Path $PSScriptRoot 'local_server.py'))
if ($NoBrowser) { $serverArgs += '--no-browser' }
& $pythonPath @serverArgs
exit $LASTEXITCODE
