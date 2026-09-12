$ErrorActionPreference = 'Stop'
Set-Location 'D:\GitHub'
$source = 'D:\GitHub\RGMT_1'
$destination = 'D:\GitHub\RGMT'
if (Test-Path -LiteralPath $source) {
    if (Test-Path -LiteralPath $destination) {
        $link = Get-Item -LiteralPath $destination -Force
        if ($link.LinkType -ne 'Junction' -or $link.Target -ne $source) {
            throw 'RGMT exists and is not the expected temporary junction.'
        }
        # Delete only the verified junction, never its target contents.
        [System.IO.Directory]::Delete($destination)
    }
    try {
        Rename-Item -LiteralPath $source -NewName 'RGMT'
    } catch {
        New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
        throw 'Directory is still in use. Close editors and terminals using RGMT_1, then retry.'
    }
}
& 'D:\miniconda3\envs\env_isaaclab\python.exe' -m pip install -e "$destination\source\RGMT" --no-deps --no-build-isolation
if ($LASTEXITCODE -ne 0) { throw 'Editable installation failed.' }
Write-Host 'Project directory and editable installation now use D:\GitHub\RGMT.'
