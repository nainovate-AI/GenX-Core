# genx_platform/genx_components/microservices/metrics/scripts/cleanup-networks.ps1
# Script to check and clean up Docker networks

Write-Host "Docker Network Cleanup Utility" -ForegroundColor Green
Write-Host "==============================" -ForegroundColor Green

# List all Docker networks
Write-Host "`nCurrent Docker networks:" -ForegroundColor Yellow
docker network ls

# Check for conflicting networks
Write-Host "`nChecking for network conflicts..." -ForegroundColor Yellow
$networks = docker network ls --format "{{.Name}}"
$genxNetworks = $networks | Where-Object { $_ -like "*genx*" -or $_ -like "*metrics*" }

if ($genxNetworks) {
    Write-Host "`nFound GenX-related networks:" -ForegroundColor Cyan
    $genxNetworks | ForEach-Object { Write-Host "  - $_" }
    
    $cleanup = Read-Host "`nDo you want to remove these networks? (y/N)"
    if ($cleanup -eq 'y') {
        foreach ($network in $genxNetworks) {
            Write-Host "Removing network: $network" -ForegroundColor Yellow
            docker network rm $network 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [OK] Removed $network" -ForegroundColor Green
            } else {
                Write-Host "  [SKIP] Could not remove $network (may be in use)" -ForegroundColor Red
            }
        }
    }
}

# Remove unused networks
Write-Host "`nRemoving unused networks..." -ForegroundColor Yellow
docker network prune -f

Write-Host "`nNetwork cleanup complete!" -ForegroundColor Green