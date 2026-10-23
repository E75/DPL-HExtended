<#
.SYNOPSIS
Puts Driv3r's Miami and its cars into your Driver Parallel Lines (DPL3 Extended hook BY E75, beta 0.1).

.DESCRIPTION
Run this script and answer the questions: it asks for the folder of Driver Parallel Lines and the folder of Driv3r
(PC, version 1.20), checks everything, converts Miami and its cars and installs the hook.

Your original game files are never changed. Cities\levels.txt is backed up to <DPL folder>\_backup\<date>\ first.
Everything that happens is written to Convert-Miami.log next to this script (and copied into the DPL folder as
DPLHookExtended.install.log): send that file when something goes wrong.

.PARAMETER DplDir
The folder that holds DriverParallelLines.exe. Asked for when left out.

.PARAMETER Driv3rDir
The folder that holds Driv3r.exe (version 1.20). Asked for when left out.

.PARAMETER SkipHook
Do not copy DPLHookExtended.asi into the DPL folder.

.PARAMETER Yes
Do not wait for Enter before the conversion starts.

.EXAMPLE
.\Convert-Miami.ps1

.EXAMPLE
.\Convert-Miami.ps1 -DplDir "D:\Games\Driver Parallel Lines" -Driv3rDir "D:\Games\Driv3r" -Yes
#>
[CmdletBinding()]
param(
	[string]$DplDir,
	[string]$Driv3rDir,
	[switch]$SkipHook,
	[switch]$Yes
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$tools = Join-Path $PSScriptRoot 'cities'
$logFile = Join-Path $PSScriptRoot 'Convert-Miami.log'
$driv3rExeSize = 4288512   # Driv3r.exe of version 1.20
$neededFreeGb = 6

Start-Transcript -LiteralPath $logFile -Force | Out-Null

function Stop-Install([string]$text) {
	Write-Host ''
	Write-Host "STOPPED: $text" -ForegroundColor Red
	Write-Host 'Run the script again after fixing this; your original game files are untouched.' -ForegroundColor Red
	Write-Host "Log file: $logFile"
	Stop-Transcript | Out-Null
	exit 1
}

function Read-GameFolder([string]$question, [string]$exeName, [string]$hint) {
	while ($true) {
		$answer = (Read-Host $question).Trim().Trim('"')

		if ($answer -and (Test-Path -LiteralPath (Join-Path $answer $exeName))) { return (Resolve-Path -LiteralPath $answer).Path }

		Write-Host "  That folder has no $exeName. $hint" -ForegroundColor Yellow
	}
}

function Write-Check([bool]$ok, [string]$passed, [string]$failed) {
	if ($ok) { Write-Host "  [ok] $passed" -ForegroundColor Green } else { Stop-Install $failed }
}

Write-Host '=== DPL3 Extended hook BY E75 (beta 0.1): Miami installer ===' -ForegroundColor Cyan
Write-Host 'Needs your own copies of Driver Parallel Lines (PC) and Driv3r (PC, version 1.20).'
Write-Host ''

if (-not $DplDir) { $DplDir = Read-GameFolder 'Folder of Driver Parallel Lines (the one with DriverParallelLines.exe)' 'DriverParallelLines.exe' 'Example: C:\GOG Games\Driver Parallel Lines' }
if (-not $Driv3rDir) { $Driv3rDir = Read-GameFolder 'Folder of Driv3r version 1.20 (the one with Driv3r.exe)' 'Driv3r.exe' 'Example: C:\Games\Driv3r' }

Write-Host ''
Write-Host 'Checking your computer and game folders...'
Write-Check (Test-Path -LiteralPath (Join-Path $DplDir 'DriverParallelLines.exe')) 'Driver Parallel Lines found' "$DplDir has no DriverParallelLines.exe."
Write-Check (Test-Path -LiteralPath (Join-Path $DplDir 'Cities\nyc_now_pc.d4c')) 'Driver Parallel Lines files complete' "$DplDir is not a complete Driver Parallel Lines folder (Cities\nyc_now_pc.d4c is missing)."
Write-Check (Test-Path -LiteralPath (Join-Path $Driv3rDir 'Driv3r.exe')) 'Driv3r found' "$Driv3rDir has no Driv3r.exe."
$exeSize = (Get-Item -LiteralPath (Join-Path $Driv3rDir 'Driv3r.exe')).Length
Write-Check ($exeSize -eq $driv3rExeSize) 'Driv3r is version 1.20' "Driv3r.exe is $exeSize bytes, version 1.20 is $driv3rExeSize. Patch Driv3r to version 1.20 and try again."
Write-Check (Test-Path -LiteralPath (Join-Path $Driv3rDir 'Anims\animation_miami.ab3')) 'Driv3r files complete' "$Driv3rDir is not a complete Driv3r folder (Anims\animation_miami.ab3 is missing)."

$running = Get-Process DriverParallelLines -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$DplDir*" }
Write-Check ($null -eq $running) 'Driver Parallel Lines is closed' 'Driver Parallel Lines is running. Close it first.'

$python = Get-Command python -ErrorAction SilentlyContinue
Write-Check ($null -ne $python) 'Python found' 'Python was not found. Install Python 3.10 or newer from python.org (tick "Add python.exe to PATH") and run this script again.'
$python = $python.Source
& $python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
Write-Check ($LASTEXITCODE -eq 0) 'Python is 3.10 or newer' 'Python is older than 3.10. Install a newer one from python.org.'
& $python -c 'import numpy, PIL' 2>$null
Write-Check ($LASTEXITCODE -eq 0) 'numpy and Pillow installed' 'Python needs two add-ons. Run: python -m pip install numpy Pillow   then run this script again.'

$drive = (Get-Item -LiteralPath $DplDir).PSDrive
$freeGb = [math]::Round($drive.Free / 1GB, 1)
Write-Check ($freeGb -ge $neededFreeGb) "$freeGb GB free on drive $($drive.Name):" "Only $freeGb GB free on drive $($drive.Name):, $neededFreeGb GB are needed."

if (-not (Test-Path -LiteralPath (Join-Path $DplDir 'DPLHook.asi'))) {
	Write-Host '  [!!] DPLHook.asi (by ermaccer, https://github.com/ermaccer/dplhook) is not in the game folder. Install it too, or the hook does not start.' -ForegroundColor Yellow
}

Write-Host ''
Write-Host "Driver Parallel Lines: $DplDir"
Write-Host "Driv3r:                $Driv3rDir"
Write-Host 'Takes about 5 to 20 minutes and about 2 GB of disk space. Your own game files stay as they are.'
if (-not $Yes) { Read-Host 'Press Enter to start (Ctrl+C to cancel)' | Out-Null }

$steps = @(
	@{ Name = 'Miami city (the long one)'; Arguments = @('d3_city_build.py', $DplDir, (Join-Path $DplDir 'Cities\miami_pc.d4c'), '--driv3r', $Driv3rDir, '--city', 'Miami', '--map', (Join-Path $DplDir 'Overlays\miami.map'), '--level') },
	@{ Name = 'HUD'; Arguments = @('d3_hud.py', $DplDir, '--level') },
	@{ Name = 'Tanner'; Arguments = @('d3_tanner.py', $DplDir, $Driv3rDir, '--level') },
	@{ Name = 'Walk, swim and climb animations'; Arguments = @('d3_anims.py', 'build', (Join-Path $Driv3rDir 'Anims\animation_miami.ab3'), (Join-Path $Driv3rDir 'Missions\mission01.dam'), (Join-Path $DplDir 'anims\nyc_now.an4'), (Join-Path $DplDir 'anims\nyc_now_d3.an4')) },
	@{ Name = 'Car plan'; Arguments = @('d3_vehicle_dlc.py', $DplDir, $Driv3rDir, 'plan', '--city', 'miami') },
	@{ Name = 'Miami cars (the second long one)'; Arguments = @('d3_vehicle_dlc.py', $DplDir, $Driv3rDir, 'build', '--city', 'miami', '--add') },
	@{ Name = 'Miami traffic'; Arguments = @('d3_vehicle_dlc.py', $DplDir, $Driv3rDir, 'traffic', '--city', 'miami') },
	@{ Name = 'Ladders'; Arguments = @('d3_ladders.py', $Driv3rDir, 'miami', (Join-Path $DplDir 'Mods\Cars\D3\miami\ladders.txt')) },
	@{ Name = 'Car handling'; Arguments = @('d3_handling_map.py', '--dpl', $DplDir, '--game', $Driv3rDir) },
	@{ Name = 'Level settings'; Arguments = @('d3_light_levels.py', $DplDir) },
	@{ Name = 'Title menu'; Arguments = @('d3_frontend_menu.py', $DplDir) }
)

New-Item -ItemType Directory -Force -Path (Join-Path $DplDir 'Mods\Cars\D3\miami') | Out-Null
$started = Get-Date

for ($i = 0; $i -lt $steps.Count; $i++) {
	$step = $steps[$i]
	$stepStart = Get-Date
	Write-Host ''
	Write-Host "[$($i + 1)/$($steps.Count)] $($step.Name)" -ForegroundColor Cyan
	$script = Join-Path $tools $step.Arguments[0]
	$rest = @($step.Arguments | Select-Object -Skip 1)

	# native output is read line by line; errors on stderr must not stop the script, the exit code decides
	$ErrorActionPreference = 'Continue'
	& $python -u $script @rest 2>&1 | ForEach-Object { Write-Host "    $_" }
	$exit = $LASTEXITCODE
	$ErrorActionPreference = 'Stop'

	if ($exit -ne 0) { Stop-Install "Step $($i + 1) ($($step.Name)) failed with code $exit. The lines above say why; send $logFile if you ask for help." }

	Write-Host ('    done in {0:N0} s' -f ((Get-Date) - $stepStart).TotalSeconds)
}

# the hook reads the Driv3r folder (handling, weather moods) from its ini
$ini = Join-Path $DplDir 'DPLHookExtended.ini'
$entry = "sDriv3rDir=$Driv3rDir"
$lines = @(if (Test-Path -LiteralPath $ini) { Get-Content -LiteralPath $ini } else { '[Extended]' })

if ($lines -match '^sDriv3rDir=') { $lines = @($lines | ForEach-Object { if ($_ -match '^sDriv3rDir=') { $entry } else { $_ } }) }
elseif ($lines -contains '[Extended]') { $lines = @($lines | ForEach-Object { $_; if ($_ -eq '[Extended]') { $entry } }) }
else { $lines = @('[Extended]', $entry) + $lines }

Set-Content -LiteralPath $ini -Value $lines -Encoding ascii
Write-Host ''
Write-Host "Driv3r folder written to $ini"

if (-not $SkipHook) {
	$asi = Join-Path $PSScriptRoot 'DPLHookExtended.asi'
	Write-Check (Test-Path -LiteralPath $asi) 'Hook file found' 'DPLHookExtended.asi is not next to this script.'
	Copy-Item -LiteralPath $asi -Destination $DplDir -Force
	Write-Host 'Hook installed: DPLHookExtended.asi is in the game folder.'
}

Write-Host ''
Write-Host ('All done in {0:N0} minutes.' -f ((Get-Date) - $started).TotalMinutes) -ForegroundColor Green
Write-Host 'Start the game, choose "D3 DLC" on the title screen, then "Miami".'
Write-Host 'The hook window opens with F2 (or Back + Start on a pad).'
Write-Host "If something is wrong, send Convert-Miami.log and the game folder's DPLHookExtended.log."
Stop-Transcript | Out-Null
Copy-Item -LiteralPath $logFile -Destination (Join-Path $DplDir 'DPLHookExtended.install.log') -Force
