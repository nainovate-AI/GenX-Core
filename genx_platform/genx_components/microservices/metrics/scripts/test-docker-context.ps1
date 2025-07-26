# genx_platform/genx_components/microservices/metrics/scripts/test-docker-context.ps1
# Test script to verify Docker build context

Write-Host "Testing Docker build context..." -ForegroundColor Yellow

# Get current location
$CurrentPath = Get-Location
Write-Host "Current directory: $CurrentPath" -ForegroundColor Cyan

# Navigate to genx_platform root
$GenxPlatformPath = (Get-Item $CurrentPath).Parent.Parent.Parent.FullName
Write-Host "GenX Platform root: $GenxPlatformPath" -ForegroundColor Cyan

# Check if Dockerfile exists
$DockerfilePath = Join-Path $CurrentPath "Dockerfile"
if (Test-Path $DockerfilePath) {
    Write-Host "✓ Dockerfile found at: $DockerfilePath" -ForegroundColor Green
} else {
    Write-Host "✗ Dockerfile not found at: $DockerfilePath" -ForegroundColor Red
}

# Check if we can access genx_platform structure
Push-Location $GenxPlatformPath
Write-Host "`nDirectory structure from genx_platform root:" -ForegroundColor Yellow
Get-ChildItem -Directory | Select-Object -First 5 | ForEach-Object { Write-Host "  - $_" }

# Check if genx_components exists
if (Test-Path "genx_components") {
    Write-Host "✓ genx_components directory found" -ForegroundColor Green
    
    # Check subdirectories
    if (Test-Path "genx_components\common") {
        Write-Host "✓ genx_components\common found" -ForegroundColor Green
    }
    if (Test-Path "genx_components\microservices\metrics") {
        Write-Host "✓ genx_components\microservices\metrics found" -ForegroundColor Green
    }
} else {
    Write-Host "✗ genx_components directory not found" -ForegroundColor Red
}

# Test Docker build command (dry run)
Write-Host "`nTesting Docker build command (dry run)..." -ForegroundColor Yellow
Write-Host "docker build --no-cache --target builder -f genx_components\microservices\metrics\Dockerfile ." -ForegroundColor Gray

Pop-Location

Write-Host "`nTo run the actual build from here, use:" -ForegroundColor Yellow
Write-Host "  .\make.ps1 build" -ForegroundColor Cyan