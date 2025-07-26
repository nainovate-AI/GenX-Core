# genx_platform/genx_components/microservices/metrics/setup.ps1
# Setup script for initial configuration

Write-Host "GenX Metrics Service - Initial Setup" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Check if .env exists
if (Test-Path ".env") {
    Write-Host " .env file already exists" -ForegroundColor Yellow
    $overwrite = Read-Host "Do you want to overwrite it? (y/N)"
    if ($overwrite -eq 'y') {
        if (Test-Path ".env.template") {
            Copy-Item ".env.template" ".env" -Force
            Write-Host " Overwrote existing .env with template" -ForegroundColor Green
        } else {
            Write-Host " .env.template not found!" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "Keeping existing .env file" -ForegroundColor Cyan
    }
} else {
    if (Test-Path ".env.template") {
        Copy-Item ".env.template" ".env" -Force
        Write-Host " Created .env from template" -ForegroundColor Green
    } else {
        Write-Host " .env.template not found!" -ForegroundColor Red
        exit 1
    }
}

# Generate secure tokens
Write-Host "`nGenerating secure tokens..." -ForegroundColor Yellow
$authToken = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
$grafanaPassword = -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 16 | ForEach-Object { [char]$_ })

# Update .env with generated values
$envContent = Get-Content ".env" -Raw
$envContent = $envContent -replace 'AUTH_TOKEN=your-secure-token-here', "AUTH_TOKEN=$authToken"
$envContent = $envContent -replace 'METRICS_AUTH_TOKEN=your-secure-token-here', "METRICS_AUTH_TOKEN=$authToken"
$envContent = $envContent -replace 'GRAFANA_PASSWORD=changeme', "GRAFANA_PASSWORD=$grafanaPassword"
Set-Content ".env" $envContent

Write-Host " Generated secure tokens" -ForegroundColor Green

# Create necessary directories
$directories = @(
    "certs",
    "logs",
    "monitoring/prometheus/alerts",
    "monitoring/grafana/provisioning/dashboards",
    "monitoring/grafana/provisioning/datasources",
    "monitoring/grafana/dashboards",
    "monitoring/otel-collector",
    "monitoring/envoy",
    "monitoring/loki",
    "monitoring/promtail",
    "monitoring/alertmanager"
)

Write-Host "`nCreating directories..." -ForegroundColor Yellow
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "   Created $dir" -ForegroundColor Gray
    }
}

# Check Docker
Write-Host "`nChecking Docker..." -ForegroundColor Yellow
try {
    docker version | Out-Null
    Write-Host " Docker is running" -ForegroundColor Green
} catch {
    Write-Host " Docker is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}

# Generate certificates if they don't exist
if (-not (Test-Path "certs/ca.crt")) {
    Write-Host "`nGenerating TLS certificates..." -ForegroundColor Yellow
    & ".\make.ps1" certs-generate
} else {
    Write-Host " TLS certificates already exist" -ForegroundColor Green
}

# Display configuration
Write-Host "`n================================" -ForegroundColor Cyan
Write-Host "Configuration Summary:" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Auth Token: $authToken" -ForegroundColor Yellow
Write-Host "Grafana Password: $grafanaPassword" -ForegroundColor Yellow
Write-Host ""
Write-Host "Important: Save these credentials securely!" -ForegroundColor Red
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Review and update .env file with your SMTP settings (optional)" -ForegroundColor White
Write-Host "2. Build the Docker image: .\make.ps1 build" -ForegroundColor White
Write-Host "3. Start the services: .\make.ps1 up-full" -ForegroundColor White
Write-Host ""
Write-Host "For development (no auth/TLS):" -ForegroundColor Yellow
Write-Host "  .\make.ps1 up-dev" -ForegroundColor White

