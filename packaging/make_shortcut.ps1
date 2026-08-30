# Create Desktop and Start Menu shortcuts for the packaged application.
param(
    [string]$Target = (Join-Path $PSScriptRoot "..\dist\RaagaComposer\RaagaComposer.exe")
)
$Target = (Resolve-Path $Target).Path
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath('Desktop'),
                   (Join-Path ([Environment]::GetFolderPath('StartMenu')) 'Programs'))) {
    $link = $shell.CreateShortcut((Join-Path $dir 'Raaga AI Music Composer.lnk'))
    $link.TargetPath = $Target
    $link.WorkingDirectory = Split-Path $Target
    $link.Description = 'Raaga-aware interactive AI music composer'
    $link.Save()
    Write-Host "Shortcut created in $dir"
}
