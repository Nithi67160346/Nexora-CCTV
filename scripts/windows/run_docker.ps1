param(
    [ValidateSet('auto', 'cpu', 'gpu')][string]$Device = 'auto',
    [ValidateSet('start', 'stop', 'status', 'logs', 'export', 'verify')][string]$Action = 'start',
    [ValidateRange(0, 65535)][int]$Port = 0,
    [switch]$Rebuild,
    [switch]$CheckOnly,
    [switch]$NoBuild,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$stateDirectory = Join-Path $projectRoot 'local_only'
$stateFile = Join-Path $stateDirectory 'docker_launcher.json'
$transcriptStarted = $false
$logFile = $null
$readyBundle = $null
$hasher = [System.Security.Cryptography.SHA256]::Create()
try {
    $rootHash = [BitConverter]::ToString($hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($projectRoot.ToLowerInvariant()))).Replace('-', '').Substring(0, 8).ToLowerInvariant()
} finally { $hasher.Dispose() }
$projectName = "nexora-qa-$rootHash"
$imageName = "nexora-qa:$Device-20261005"

function Invoke-QaDocker {
    param([string[]]$Arguments)
    & $script:dockerCommand.Source @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed ($LASTEXITCODE): docker $($Arguments -join ' ')" }
}

function Find-QaPort {
    param([int]$Preferred)
    for ($candidate = $Preferred; $candidate -le [Math]::Min(65535, $Preferred + 20); $candidate++) {
        $socket = New-Object System.Net.Sockets.Socket([Net.Sockets.AddressFamily]::InterNetwork, [Net.Sockets.SocketType]::Stream, [Net.Sockets.ProtocolType]::Tcp)
        try {
            $socket.ExclusiveAddressUse = $true
            $socket.Bind((New-Object Net.IPEndPoint([Net.IPAddress]::Loopback, $candidate)))
            return $candidate
        } catch [Net.Sockets.SocketException] {
            if ($_.Exception.NativeErrorCode -notin @(10048, 10013)) { throw }
        } finally { $socket.Dispose() }
    }
    throw "No available port in $Preferred..$([Math]::Min(65535, $Preferred + 20)). Use -Port with another range."
}

function Get-QaFileHash {
    param([string]$Path)
    $algorithm = [Security.Cryptography.SHA256]::Create()
    $stream = [IO.File]::OpenRead($Path)
    try { return [BitConverter]::ToString($algorithm.ComputeHash($stream)).Replace('-', '').ToLowerInvariant() }
    finally { $stream.Dispose(); $algorithm.Dispose() }
}

function Get-QaWebRevision {
    param([string]$Target)
    $labelJson = & $script:dockerCommand.Source inspect --format '{{json .Config.Labels}}' $Target
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the QA web revision.' }
    $labels = ($labelJson -join "`n") | ConvertFrom-Json
    return [string]$labels.'nexora.web.revision'
}

function Join-QaReadyArchive {
    param([string]$Destination)
    if (-not $readyBundle.archive_parts) { throw 'Ready image TAR is missing. Download all image parts from the matching Release.' }
    $partPaths = @()
    $seen = @{}
    foreach ($part in $readyBundle.archive_parts) {
        if ($part.name -notmatch '^nexora-qa-ready-[a-z0-9-]+\.tar\.part[0-9]{3}$' -or
            $part.sha256 -notmatch '^[a-f0-9]{64}$' -or $part.bytes -le 0 -or $seen.ContainsKey($part.name)) {
            throw 'Invalid ready image part manifest.'
        }
        $seen[$part.name] = $true
        $found = $null
        # Windows Extract All adds an outer folder around the ZIP's own folder.
        $parentFolder = Split-Path -Parent $projectRoot
        foreach ($folder in @((Join-Path $projectRoot 'docker_images'), $projectRoot, $parentFolder, (Split-Path -Parent $parentFolder))) {
            $candidate = Join-Path $folder $part.name
            if (Test-Path -LiteralPath $candidate -PathType Leaf) { $found = $candidate; break }
        }
        if (-not $found) { throw "Missing image part: $($part.name). Download every part into the same folder as the QA ZIP." }
        if ((Get-Item -LiteralPath $found).Length -ne $part.bytes -or (Get-QaFileHash -Path $found) -ne $part.sha256) {
            throw "Image part checksum mismatch: $($part.name). Download this part again; no image was loaded."
        }
        $partPaths += $found
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $partial = $Destination + '.joining'
    Write-Host 'Combining verified image parts. Keep this window open.' -ForegroundColor Cyan
    $output = [IO.File]::Open($partial, [IO.FileMode]::Create, [IO.FileAccess]::Write, [IO.FileShare]::None)
    try {
        foreach ($path in $partPaths) {
            $inputStream = [IO.File]::OpenRead($path)
            try { $inputStream.CopyTo($output) } finally { $inputStream.Dispose() }
        }
    } finally { $output.Dispose() }
    if ((Get-QaFileHash -Path $partial) -ne $readyBundle.archive_sha256) {
        throw 'Combined image checksum mismatch. No image was loaded; download the matching Release parts again.'
    }
    Move-Item -LiteralPath $partial -Destination $Destination
}

function Get-QaSourceFingerprint {
    # Only files copied into the image. Model mounts and QA/user data are excluded.
    $sourceFiles = @()
    foreach ($folder in @('app', 'integration', 'web', 'Location/location', 'Wandering/ai_camera_system', 'docker/assets')) {
        $path = Join-Path $projectRoot $folder
        if (Test-Path -LiteralPath $path -PathType Container) {
            $sourceFiles += Get-ChildItem -LiteralPath $path -File -Recurse | Where-Object {
                $_.FullName -notmatch '[\\/](tests|__pycache__|node_modules)[\\/]' -and
                $_.FullName -notmatch '[\\/]app[\\/]features[\\/]seizure[\\/]' -and
                $_.Name -notlike '.env*' -and
                $_.Name -notmatch '\.(pt|pth|pkl|onnx|joblib|npz|mp4|avi|mov|mkv|webm|zip|tar|tar\.gz)$'
            }
        }
    }
    foreach ($file in @('Dockerfile','Fall/Project.py','docker/requirements.lock.txt','docker/torch.constraints.txt','docker/entrypoint.py','docker/healthcheck.py')) {
        $path = Join-Path $projectRoot $file
        if (Test-Path -LiteralPath $path -PathType Leaf) { $sourceFiles += Get-Item -LiteralPath $path }
    }
    $manifest = @($sourceFiles | Sort-Object FullName | ForEach-Object {
        $_.FullName.Substring($projectRoot.Length + 1).Replace('\', '/') + ':' + (Get-QaFileHash -Path $_.FullName)
    }) -join "`n"
    $algorithm = [Security.Cryptography.SHA256]::Create()
    try { return [BitConverter]::ToString($algorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($manifest))).Replace('-', '').ToLowerInvariant() }
    finally { $algorithm.Dispose() }
}

function Test-QaImageCurrent {
    param([string]$Target)
    if ((Get-QaWebRevision -Target $Target) -ne $expectedWebRevision) { return $false }
    $labelJson = & $script:dockerCommand.Source inspect --format '{{json .Config.Labels}}' $Target
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the QA source fingerprint.' }
    $labels = ($labelJson -join "`n") | ConvertFrom-Json
    return [string]$labels.'nexora.source.fingerprint' -eq $expectedSourceFingerprint
}

function Test-QaGpuImage {
    # No pulls, build, mounted files, network or inference. Probe only a local GPU image.
    $gpuProbeImage = 'nexora-qa:gpu-20261005'
    if ($readyBundle) { $gpuProbeImage = $readyBundle.image }
    $images = @(& $script:dockerCommand.Source image ls --quiet --filter "reference=$gpuProbeImage")
    if ($LASTEXITCODE -ne 0 -or -not @($images | Where-Object { $_.ToString().Trim() }).Count) {
        Write-Host 'No local GPU image to verify. Using CPU; use -Device gpu to build the GPU image when ready.' -ForegroundColor Yellow
        return $false
    }
    $previousPreference = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
    try {
        $output = @(& $script:dockerCommand.Source run --rm --pull never --network none --gpus all --entrypoint python $gpuProbeImage -c 'import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0))' 2>&1)
        $probeExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previousPreference }
    if ($probeExit -ne 0) {
        Write-Host "Docker GPU is unavailable; using CPU. Check NVIDIA driver and Docker Desktop WSL2 backend.`n$($output -join "`n")" -ForegroundColor Yellow
        return $false
    }
    Write-Host ("Docker CUDA ready: " + ($output -join ', ')) -ForegroundColor Cyan
    return $true
}

try {
    if ($Action -eq 'start' -and -not $CheckOnly) {
        $logDirectory = Join-Path $projectRoot 'local_only\qa'
        New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
        $logFile = Join-Path $logDirectory ("docker_start_{0}_{1}.log" -f $Device, (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
        Start-Transcript -LiteralPath $logFile | Out-Null
        $transcriptStarted = $true
        Write-Host "Startup/build log: $logFile"
    }
    $dockerCommand = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $dockerCommand) {
        foreach ($dockerPath in @("$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe", "$env:ProgramFiles\Docker\Docker\resources\bin\docker.exe")) {
            if (Test-Path -LiteralPath $dockerPath -PathType Leaf) { $dockerCommand = Get-Command $dockerPath; break }
        }
    }
    if (-not $dockerCommand) { throw 'Docker is not installed. Install Docker Desktop and open it in Linux containers mode. See QA_DOCKER_TH.md.' }
    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $engine = & $dockerCommand.Source version --format '{{.Server.Os}}' 2>&1 }
    finally { $ErrorActionPreference = $previousErrorPreference }
    if ($LASTEXITCODE -ne 0) { throw "Docker Engine is not ready. Open Docker Desktop, wait until Engine running, then retry.`n$($engine -join "`n")" }
    if (($engine | Select-Object -Last 1).ToString().Trim() -ne 'linux') { throw 'Select Linux containers in Docker Desktop; Windows containers cannot run this image.' }
    $engineArch = & $dockerCommand.Source version --format '{{.Server.Arch}}'
    if ($LASTEXITCODE -ne 0 -or ($engineArch | Select-Object -Last 1).ToString().Trim() -ne 'amd64') {
        throw 'This QA image requires a Linux amd64 engine (Intel/AMD x64). Mac/Windows ARM needs a separately tested image.'
    }
    Invoke-QaDocker -Arguments @('compose', 'version')

    $readyFile = Join-Path $projectRoot 'QA_READY.json'
    if (Test-Path -LiteralPath $readyFile -PathType Leaf) {
        $readyBundle = Get-Content -LiteralPath $readyFile -Raw -Encoding UTF8 | ConvertFrom-Json
        if ($readyBundle.image -notmatch '^nexora-qa:ready-[a-z0-9-]+$' -or
            $readyBundle.image_id -notmatch '^sha256:[a-f0-9]{64}$' -or
            $readyBundle.archive -notmatch '^nexora-qa-ready-[a-z0-9-]+\.tar$' -or
            $readyBundle.source_fingerprint -notmatch '^[a-f0-9]{64}$' -or
            $readyBundle.archive_sha256 -notmatch '^[a-f0-9]{64}$') { throw 'Invalid QA_READY.json. Extract the complete ready-to-run QA package again.' }
        if (-not $readyBundle.files) { throw 'Ready package file manifest is missing.' }
        foreach ($readyItem in $readyBundle.files) {
            $readyPath = [IO.Path]::GetFullPath((Join-Path $projectRoot $readyItem.path))
            if (-not $readyPath.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                $readyItem.sha256 -notmatch '^[a-f0-9]{64}$' -or -not (Test-Path -LiteralPath $readyPath -PathType Leaf)) { throw 'Ready package files are incomplete. Extract the whole QA ZIP again.' }
            if ((Get-QaFileHash -Path $readyPath) -ne $readyItem.sha256) { throw "Ready package file checksum mismatch: $($readyItem.path). Extract the complete QA ZIP again." }
        }
        if ($Rebuild) { throw 'This ready-to-run package never builds. Obtain a newer QA package, or rebuild in the development project.' }
        $env:NEXORA_QA_IMAGE = [string]$readyBundle.image
        if ($Action -eq 'start' -and -not $CheckOnly) {
            $readyImages = @(& $dockerCommand.Source image ls --quiet --filter "reference=$($readyBundle.image)")
            if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the ready-to-run image.' }
            if (-not @($readyImages | Where-Object { $_.ToString().Trim() }).Count) {
                if ($NoBuild) { throw 'NoBuild requires the ready image already loaded; double-click start_docker.cmd to import it first.' }
                $readyArchive = Join-Path (Join-Path $projectRoot 'docker_images') $readyBundle.archive
                if (-not (Test-Path -LiteralPath $readyArchive -PathType Leaf)) { Join-QaReadyArchive -Destination $readyArchive }
                Write-Host 'Checking the ready image, then importing it. First import can take several minutes; no build or download.' -ForegroundColor Cyan
                if ((Get-QaFileHash -Path $readyArchive) -ne $readyBundle.archive_sha256) { throw 'Ready image TAR checksum mismatch. No image was loaded.' }
                Invoke-QaDocker -Arguments @('load','--input',$readyArchive)
            }
            $readyImageId = & $dockerCommand.Source image inspect $readyBundle.image --format '{{.Id}}'
            if ($LASTEXITCODE -ne 0 -or ($readyImageId -join '').Trim() -ne $readyBundle.image_id) { throw 'The ready image does not match this QA package. Obtain the complete matching package.' }
        }
    } else { Remove-Item Env:NEXORA_QA_IMAGE -ErrorAction SilentlyContinue }

    if (Test-Path -LiteralPath $stateFile -PathType Leaf) {
        try { $saved = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json }
        catch { $saved = $null; Write-Host 'Saved launcher settings are unreadable; checking the actual container instead.' -ForegroundColor Yellow }
        if ($Action -ne 'start' -and ($Device -eq 'auto' -or -not $PSBoundParameters.ContainsKey('Device')) -and $saved.device -in @('cpu', 'gpu')) {
            $Device = $saved.device
            $imageName = "nexora-qa:$Device-20261005"
        }
        if ($Port -eq 0 -and $saved.port -ge 1 -and $saved.port -le 65535) { $Port = [int]$saved.port }
    }
    if ($Device -eq 'auto') {
        $Device = 'cpu'
        if ($Action -eq 'start') {
            # Reopening an active session must not switch device or lose its reviews.
            $existingArgs = @('compose','--project-name',$projectName,'--project-directory',$projectRoot,'-f',(Join-Path $projectRoot 'compose.yaml'))
            $existingIds = @(& $dockerCommand.Source @existingArgs ps --all -q nexora)
            if ($LASTEXITCODE -ne 0) { throw 'Could not inspect this QA project before device selection.' }
            $keepDevice = $false
            if ($existingIds.Count -eq 1 -and $existingIds[0]) {
                $detailsJson = & $dockerCommand.Source inspect $existingIds[0].Trim()
                if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the existing QA session.' }
                $details = ($detailsJson -join "`n") | ConvertFrom-Json
                if ($details[0].State.Running) {
                    if ($details[0].Config.Image -eq 'nexora-qa:gpu-20261005') { $Device = 'gpu'; $keepDevice = $true }
                    elseif ($details[0].Config.Image -eq 'nexora-qa:cpu-20261005') { $keepDevice = $true }
                    elseif ($readyBundle -and $details[0].Config.Image -eq $readyBundle.image) {
                        $Device = if ($details[0].Config.Env -contains 'NEXORA_DEVICE=cuda') { 'gpu' } else { 'cpu' }
                        $keepDevice = $true
                    }
                }
            }
            $nvidiaCommand = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($nvidiaCommand -and -not $keepDevice) {
                $previousErrorPreference = $ErrorActionPreference
                $ErrorActionPreference = 'Continue'
                try { $gpuNames = @(& $nvidiaCommand.Source --query-gpu=name --format=csv,noheader 2>&1); $gpuProbeExit = $LASTEXITCODE }
                finally { $ErrorActionPreference = $previousErrorPreference }
                if ($gpuProbeExit -eq 0 -and @($gpuNames | Where-Object { $_.ToString().Trim() }).Count -gt 0) {
                    Write-Host ("Detected NVIDIA GPU: " + ($gpuNames -join ', ')) -ForegroundColor Cyan
                    if ($CheckOnly) {
                        Write-Host 'CheckOnly leaves CPU selected; Docker GPU requires a runtime probe on normal start.' -ForegroundColor Yellow
                    } elseif (Test-QaGpuImage) { $Device = 'gpu' }
                }
            }
            Write-Host "Automatic device selection: $Device. Use -Device cpu or -Device gpu to select explicitly." -ForegroundColor Cyan
        }
    }
    $imageName = "nexora-qa:$Device-20261005"
    if ($readyBundle) { $imageName = $readyBundle.image }
    if ($Port -eq 0) { $Port = 8002 }
    $env:NEXORA_PORT = [string]$Port
    $composeArgs = @('compose', '--progress', 'plain', '--project-name', $projectName, '--project-directory', $projectRoot, '-f', (Join-Path $projectRoot 'compose.yaml'))
    if ($Device -eq 'gpu') { $composeArgs += @('-f', (Join-Path $projectRoot 'compose.gpu.yaml')) }
    Invoke-QaDocker -Arguments ($composeArgs + @('config', '--quiet'))
    Write-Host "NEXORA QA Docker | device=$Device | project=$projectName" -ForegroundColor Cyan
    if ($Action -eq 'status') { Invoke-QaDocker -Arguments ($composeArgs + @('ps')); exit 0 }
    if ($Action -eq 'logs') { Invoke-QaDocker -Arguments ($composeArgs + @('logs', '--tail', '100', 'nexora')); exit 0 }
    if ($Action -eq 'stop') {
        if (-not $CheckOnly) { Invoke-QaDocker -Arguments ($composeArgs + @('stop', 'nexora')) }
        Write-Host 'Stopped this QA project only. Recordings and volumes were kept.'
        exit 0
    }
    $versionFile = Join-Path $projectRoot 'web\runtime_version.json'
    $expectedWebRevision = (Get-Content -LiteralPath $versionFile -Raw -Encoding UTF8 | ConvertFrom-Json).revision
    if (-not $expectedWebRevision) { throw 'Missing web revision in web\runtime_version.json. Extract the current QA ZIP.' }
    $env:NEXORA_WEB_REVISION = $expectedWebRevision
    $expectedSourceFingerprint = Get-QaSourceFingerprint
    if ($readyBundle) {
        # Ready packages execute immutable code inside the supplied image, never host source/build files.
        $expectedSourceFingerprint = $readyBundle.source_fingerprint
        if ($readyBundle.web_revision -ne $expectedWebRevision) { throw 'Ready package web revision does not match. Extract the complete QA ZIP again.' }
    }
    $env:NEXORA_SOURCE_FINGERPRINT = $expectedSourceFingerprint
    $weights = Join-Path $projectRoot 'models\yolo26n-pose.pt'
    if (-not (Test-Path -LiteralPath $weights -PathType Leaf) -or (Get-Item -LiteralPath $weights).Length -eq 0) {
        throw 'Missing models\yolo26n-pose.pt. Extract the complete QA ZIP. GitHub Source code does not contain weights.'
    }
    $interactionWeights = Join-Path $projectRoot 'best_lstm_model.pth'
    if (-not (Test-Path -LiteralPath $interactionWeights -PathType Leaf) -or (Get-Item -LiteralPath $interactionWeights).Length -eq 0) {
        throw 'Missing best_lstm_model.pth for the LSTM clip test. Extract the complete current QA ZIP.'
    }
    foreach ($fallName in @('ai1_isolation_forest_yolo.pkl','ai1_scaler_yolo.pkl','ai1_threshold_yolo.pkl','ai2_fall_detector_yolo.pkl','feature_names_yolo.pkl')) {
        $fallWeights = Join-Path $projectRoot ("Fall\models_yolo\" + $fallName)
        if (-not (Test-Path -LiteralPath $fallWeights -PathType Leaf) -or (Get-Item -LiteralPath $fallWeights).Length -eq 0) {
            throw "Missing Fall\models_yolo\$fallName. Extract the current QA ZIP with all five fall models."
        }
    }
    if ($CheckOnly) { Write-Host 'Docker / Linux / Compose / model checks passed. No build or container start was performed.'; exit 0 }

    if ($Action -eq 'verify') {
        $ids = @(& $dockerCommand.Source @composeArgs ps --all -q nexora)
        if ($LASTEXITCODE -ne 0 -or $ids.Count -ne 1 -or -not $ids[0]) { throw 'No QA container in this folder. Run start_docker.cmd first. Verification does not build or start containers.' }
        $containerId = $ids[0].Trim()
        if (-not (Test-QaImageCurrent -Target $containerId)) { throw 'The running QA image does not match the current source. Export reviews, stop this QA project and run start_docker.cmd before verification.' }
        $details = & $dockerCommand.Source inspect $containerId | ConvertFrom-Json
        if ($LASTEXITCODE -ne 0 -or -not $details[0].State.Running) { throw 'QA container is stopped. Verification does not start it.' }
        $details = $details[0]
        $Port = [int]$details.NetworkSettings.Ports.'8080/tcp'[0].HostPort
        $url = "http://127.0.0.1:$Port"
        $modelsMount = @($details.Mounts | Where-Object { $_.Destination -eq '/app/models' })
        $fallMount = @($details.Mounts | Where-Object { $_.Destination -eq '/app/Fall/models_yolo' })
        $dataMount = @($details.Mounts | Where-Object { $_.Destination -eq '/app/local_only' })
        if ($modelsMount.Count -ne 1 -or $modelsMount[0].RW -or $dataMount.Count -ne 1 -or $dataMount[0].Type -ne 'volume' -or -not $dataMount[0].RW) { throw 'Expected read-only model bind and writable data volume are missing.' }
        if ($fallMount.Count -ne 1 -or $fallMount[0].RW) { throw 'Expected read-only fall model bind is missing. Update the Docker image and compose mounts.' }
        $hostStatus = Invoke-RestMethod -Uri "$url/api/status" -TimeoutSec 10
        if ($hostStatus.health.detector.state -ne 'ready') { throw 'Published web API detector is not ready.' }
        # Run the current probe through stdin, including on images built before this helper existed.
        $probe = Get-Content -LiteralPath (Join-Path $projectRoot 'docker\smoke.py') -Raw -Encoding UTF8
        $probeOutput = $probe | & $dockerCommand.Source exec -i $containerId python -
        $probeExit = $LASTEXITCODE
        $report = ($probeOutput -join "`n") | ConvertFrom-Json
        $report | Add-Member -NotePropertyName container_image_id -NotePropertyValue $details.Image
        $report | Add-Member -NotePropertyName container_id -NotePropertyValue $containerId
        $report | Add-Member -NotePropertyName url -NotePropertyValue $url
        $report | Add-Member -NotePropertyName models_read_only -NotePropertyValue $true
        $report | Add-Member -NotePropertyName data_volume -NotePropertyValue $dataMount[0].Name
        $reportDirectory = Join-Path $projectRoot 'local_only\qa'
        New-Item -ItemType Directory -Force -Path $reportDirectory | Out-Null
        $reportFile = Join-Path $reportDirectory ("docker_runtime_{0}_{1}.json" -f $Device, (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
        $report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $reportFile -Encoding UTF8
        Write-Host "Docker runtime report: $reportFile"
        if ($probeExit -ne 0 -or -not $report.passed) { throw 'Docker runtime verification failed. Read the report checks/errors.' }
        Write-Host "Docker runtime checks passed: $url. See the report for any skipped inference and remaining manual QA." -ForegroundColor Green
        exit 0
    }

    if ($Action -eq 'export') {
        Invoke-QaDocker -Arguments @('image', 'inspect', $imageName, '--format', '{{.Id}}')
        if (-not (Test-QaImageCurrent -Target $imageName)) { throw 'The QA image is older than the current source. Run start_docker.cmd to update it before exporting for QA.' }
        $exportDirectory = Join-Path $projectRoot 'local_only\qa\docker_images'
        New-Item -ItemType Directory -Force -Path $exportDirectory | Out-Null
        $exportFile = Join-Path $exportDirectory "nexora-qa-$Device-20261005.tar"
        if (Test-Path -LiteralPath $exportFile) { throw "Image export already exists: $exportFile. Move the previous export before exporting again." }
        Write-Host 'Exporting the built image. This may take several minutes.' -ForegroundColor Cyan
        Invoke-QaDocker -Arguments @('save', '--output', $exportFile, $imageName)
        $checksum = Get-QaFileHash -Path $exportFile
        [IO.File]::WriteAllText($exportFile + '.sha256', "$checksum  $([IO.Path]::GetFileName($exportFile))`n")
        Write-Host "Image ready: $exportFile`nSend the QA ZIP and this TAR plus checksum to QA. Place the TAR in docker_images beside start_docker.cmd." -ForegroundColor Green
        exit 0
    }

    $containerIds = @(& $dockerCommand.Source @composeArgs ps --all -q nexora)
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect this QA project.' }
    $running = $false
    if ($containerIds.Count -gt 0 -and $containerIds[0]) {
        $containerId = $containerIds[0].Trim()
        $runningState = & $dockerCommand.Source inspect --format '{{.State.Running}}' $containerId
        if ($LASTEXITCODE -ne 0) { throw 'Could not inspect this QA container.' }
        $running = ($runningState | Select-Object -Last 1).Trim() -eq 'true'
        if ($running) {
            $actualImage = & $dockerCommand.Source inspect --format '{{.Config.Image}}' $containerId
            if ($LASTEXITCODE -ne 0) { throw 'Could not read the running QA image.' }
            if (($actualImage | Select-Object -Last 1).Trim() -ne $imageName -or $Rebuild) {
                throw 'A QA session is already running. Export reviews and use stop_docker.cmd before changing device or rebuilding.'
            }
            if (-not (Test-QaImageCurrent -Target $containerId)) {
                throw "This running Docker web is older than the source ($expectedWebRevision). Export reviews/profiles from the web, run stop_docker.cmd, then start_docker.cmd. The next start rebuilds the old image automatically; volumes are kept."
            }
            # Windows PowerShell 5 strips embedded native-argument quotes.
            # Read JSON instead of quoting the 8080/tcp key in a Go template.
            $portJson = & $dockerCommand.Source inspect --format '{{json .NetworkSettings.Ports}}' $containerId
            if ($LASTEXITCODE -ne 0) { throw 'Could not read the current QA port.' }
            $portBindings = ($portJson -join "`n") | ConvertFrom-Json
            $localhostBindings = @($portBindings.'8080/tcp' | Where-Object { $_.HostIp -eq '127.0.0.1' })
            if ($localhostBindings.Count -ne 1 -or $localhostBindings[0].HostPort -notmatch '^\d+$') { throw 'Could not read the current QA localhost port.' }
            $Port = [int]$localhostBindings[0].HostPort
            Write-Host 'Using the existing QA container; current reviews were kept.'
        }
    }
    if (-not $running) {
        $Port = Find-QaPort -Preferred $Port
        $localImages = @(& $dockerCommand.Source image ls --quiet --filter "reference=$imageName")
        if ($LASTEXITCODE -ne 0) { throw 'Could not check local Docker images.' }
        $imagePresent = @($localImages | Where-Object { $_.Trim() }).Count -gt 0
        $imageArchive = Join-Path $projectRoot "docker_images\nexora-qa-$Device-20261005.tar"
        if (-not $imagePresent -and -not $Rebuild -and (Test-Path -LiteralPath $imageArchive -PathType Leaf)) {
            if ($NoBuild) { throw 'NoBuild requires an image already loaded locally. Loading the TAR can take several minutes; run start_docker.cmd yourself first.' }
            $checksumFile = $imageArchive + '.sha256'
            if (-not (Test-Path -LiteralPath $checksumFile -PathType Leaf)) { throw 'Image TAR has no SHA256 file. Copy its .tar.sha256 companion before loading.' }
            $expected = ((Get-Content -LiteralPath $checksumFile -Raw).Trim() -split '\s+')[0]
            if ($expected -notmatch '^[a-fA-F0-9]{64}$' -or (Get-QaFileHash -Path $imageArchive) -ne $expected) { throw 'Image TAR checksum mismatch. No image was loaded.' }
            Write-Host 'Loading the supplied QA image; no dependency installation is needed.' -ForegroundColor Cyan
            Invoke-QaDocker -Arguments @('load', '--input', $imageArchive)
            Invoke-QaDocker -Arguments @('image', 'inspect', $imageName, '--format', '{{.Id}}')
            $imagePresent = $true
        }
        if (-not $imagePresent -or $Rebuild) {
            if ($readyBundle) { throw 'The ready image is missing. Extract the complete QA package again. No build was attempted.' }
            if ($NoBuild) { throw 'No local QA image (or Rebuild requested). No build/start was performed. Run start_docker.cmd yourself first.' }
            Write-Host 'Building the QA image. First build downloads dependencies and can take several minutes. Keep this window open.' -ForegroundColor Cyan
            Invoke-QaDocker -Arguments ($composeArgs + @('build', 'nexora'))
        } elseif (-not (Test-QaImageCurrent -Target $imageName)) {
            if ($readyBundle) { throw 'Ready image and source files do not match. Obtain the complete matching QA package. No build was attempted.' }
            if ($NoBuild) { throw 'The local QA image contains an older web. No build/start was performed. Run start_docker.cmd yourself to update.' }
            Write-Host "Updating the QA image to $expectedWebRevision; dependency cache will be reused." -ForegroundColor Cyan
            Invoke-QaDocker -Arguments ($composeArgs + @('build', 'nexora'))
        }
    }
    $env:NEXORA_PORT = [string]$Port
    Write-Host "Starting NEXORA and waiting for AI readiness: http://127.0.0.1:$Port" -ForegroundColor Cyan
    try {
        $startArguments = @('up', '-d', '--no-build', '--pull', 'never', '--wait', '--wait-timeout', '240', 'nexora')
        if ($running) { $startArguments += '--no-recreate' }
        Invoke-QaDocker -Arguments ($composeArgs + $startArguments)
    } catch {
        & $dockerCommand.Source @composeArgs logs --tail 80 nexora
        throw
    }
    New-Item -ItemType Directory -Force -Path $stateDirectory | Out-Null
    @{ device = $Device; port = $Port; project = $projectName; web_revision = $expectedWebRevision; source_fingerprint = $expectedSourceFingerprint } | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8
    $url = "http://127.0.0.1:$Port"
    Write-Host "NEXORA main web ($expectedWebRevision): $url`nUse stop_docker.cmd to stop. Closing this window does not stop Docker." -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $url -WindowStyle Hidden }
    exit 0
} catch {
    Write-Host "`nNEXORA Docker could not start:`n$($_.Exception.Message)`nSee QA_DOCKER_TH.md. No other services or volumes were removed." -ForegroundColor Red
    if ($logFile) { Write-Host "Send this full log when reporting the problem: $logFile" -ForegroundColor Yellow }
    exit 1
} finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
