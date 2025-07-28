# genx_platform/monitoring.ps1
# Platform-wide monitoring stack management script
# This script manages the monitoring infrastructure shared by all GenX microservices

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [string]$Environment = "production",
    [switch]$Detach = $true,
    [switch]$ForceRecreate = $false,
    [switch]$RemoveVolumes = $false,
    [switch]$UseTLS = $false,
    [int]$Timeout = 300
)

# Script configuration
$ErrorActionPreference = "Stop"
$Platform = "GenX Platform Monitoring"
$Version = "1.0.0"

# Function to print colored output
function Write-ColorOutput {
    param (
        [string]$Message,
        [string]$Level = "Info"
    )
    
    $color = switch ($Level) {
        "Success" { "Green" }
        "Error"   { "Red" }
        "Warning" { "Yellow" }
        "Info"    { "Cyan" }
        "Header"  { "Magenta" }
        default   { "White" }
    }
    
    Write-Host $Message -ForegroundColor $color
}

# Function to print headers
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-ColorOutput ("=" * 80) "Header"
    Write-ColorOutput " $Text" "Header"
    Write-ColorOutput ("=" * 80) "Header"
    Write-Host ""
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." "Info"
    
    # Check Docker
    try {
        $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
        if ($dockerVersion) {
            Write-ColorOutput "   Docker version: $dockerVersion" "Success"
        } else {
            throw "Docker is not running"
        }
    } catch {
        Write-ColorOutput "   Docker is not installed or not running" "Error"
        return $false
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker-compose version --short 2>$null
        if ($composeVersion) {
            Write-ColorOutput "   Docker Compose version: $composeVersion" "Success"
        } else {
            throw "Docker Compose not found"
        }
    } catch {
        Write-ColorOutput "   Docker Compose is not installed" "Error"
        return $false
    }
    
    # Check if running from correct directory
    if (-not (Test-Path "monitoring/docker-compose.monitoring.yml")) {
        Write-ColorOutput "   Not in GenX platform root directory" "Error"
        Write-ColorOutput "    Please run this script from the genx_platform root directory" "Warning"
        return $false
    }
    
    Write-ColorOutput "   Running from correct directory" "Success"
    
    # Check network
    $networkExists = docker network ls --format "{{.Name}}" | Where-Object { $_ -eq "genx-platform-network" }
    if (-not $networkExists) {
        Write-ColorOutput "  ! Platform network does not exist, will create it" "Warning"
    } else {
        Write-ColorOutput "   Platform network exists" "Success"
    }
    
    return $true
}

# Function to create platform network
function Initialize-PlatformNetwork {
    Write-ColorOutput "Initializing platform network..." "Info"
    $networkExists = docker network ls --format "{{.Name}}" | Where-Object { $_ -eq "genx-platform-network" }
    if (-not $networkExists) {
        try {
            docker network create genx-platform-network | Out-Null
            Write-ColorOutput "   Created genx-platform-network" "Success"
        } catch {
            Write-ColorOutput "   Failed to create platform network" "Error"
            return $false
        }
    }
    return $true
}

# Function to check service health
function Test-ServiceHealth {
    param(
        [string]$ServiceName,
        [string]$HealthUrl,
        [int]$MaxRetries = 30,
        [switch]$IgnoreSSL = $false
    )

    Write-ColorOutput "  Checking $ServiceName..." "Info"
    
    # For TLS endpoints, we'll use curl instead of Invoke-WebRequest to avoid SSL issues
    if ($IgnoreSSL -and $HealthUrl.StartsWith("https")) {
        for ($i = 1; $i -le $MaxRetries; $i++) {
            try {
                $result = & curl -k -s -o /dev/null -w "%{http_code}" $HealthUrl 2>$null
                if ($result -eq "200") {
                    Write-ColorOutput "    [+] $ServiceName is healthy" "Success"
                    return $true
                }
            } catch {
                # Service not ready yet
            }
            
            if ($i -lt $MaxRetries) {
                Write-Host "    Waiting for $ServiceName to be ready... ($i/$MaxRetries)" -NoNewline
                Start-Sleep -Seconds 2
                Write-Host "`r" -NoNewline
            }
        }
    } else {
        # Non-TLS health check
        for ($i = 1; $i -le $MaxRetries; $i++) {
            try {
                $response = Invoke-WebRequest -Uri $HealthUrl -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    Write-ColorOutput "    [+] $ServiceName is healthy" "Success"
                    return $true
                }
            } catch {
                # Service not ready yet
            }
            
            if ($i -lt $MaxRetries) {
                Write-Host "    Waiting for $ServiceName to be ready... ($i/$MaxRetries)" -NoNewline
                Start-Sleep -Seconds 2
                Write-Host "`r" -NoNewline
            }
        }
    }
    
    Write-ColorOutput "    [-] $ServiceName failed to become healthy" "Error"
    return $false
}

# Function to start monitoring stack
function Start-Monitoring {
    Write-Header "Starting Platform Monitoring Stack"
    
    if (-not (Test-Prerequisites)) {
        Write-ColorOutput "Prerequisites check failed. Exiting." "Error"
        exit 1
    }
    
    if (-not (Initialize-PlatformNetwork)) {
        Write-ColorOutput "Failed to initialize platform network. Exiting." "Error"
        exit 1
    }
    
    # Check certificates if using TLS
    if ($UseTLS) {
        Write-ColorOutput "TLS mode enabled. Checking certificates..." "Info"
        if (-not (Test-Path "./infrastructure/certs/ca/ca.crt")) {
            Write-ColorOutput "Certificates not found! Run .\certificates.ps1 create first." "Error"
            exit 1
        }
        Write-ColorOutput "  [+] Certificates found" "Success"
    }
    
    # Set environment variables
    $env:ENVIRONMENT = $Environment
    
    Write-ColorOutput "Starting monitoring services..." "Info"
    
    # Build command
    if ($UseTLS) {
        $composeFile = "monitoring/docker-compose.monitoring.tls.yml"
        Write-ColorOutput "Using TLS-enabled configuration" "Info"
    } else {
        $composeFile = "monitoring/docker-compose.monitoring.yml"
        Write-ColorOutput "Using standard configuration (no TLS)" "Warning"
    }
    
    $composeCmd = "docker-compose -f $composeFile up"
    
    if ($Detach) {
        $composeCmd += " -d"
    }
    
    if ($ForceRecreate) {
        $composeCmd += " --force-recreate"
    }
    
    # Start services
    Invoke-Expression $composeCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to start monitoring stack" "Error"
        exit 1
    }
    
    if ($Detach) {
        Write-ColorOutput "`nWaiting for services to be ready..." "Info"
        
        # Health checks
        $protocol = if ($UseTLS) { "https" } else { "http" }
        $healthChecks = @(
            @{Name="Prometheus"; Url="$protocol`://localhost:9090/-/ready"},
            @{Name="Grafana"; Url="$protocol`://localhost:3000/api/health"},
            @{Name="Jaeger"; Url="http://localhost:16686/"},
            @{Name="Loki"; Url="http://localhost:3100/ready"},
            @{Name="AlertManager"; Url="$protocol`://localhost:9093/-/ready"}
        )
        
        $allHealthy = $true
        foreach ($check in $healthChecks) {
            $ignoreSSL = $UseTLS -and $check.Url.StartsWith("https")
            if (-not (Test-ServiceHealth -ServiceName $check.Name -HealthUrl $check.Url -IgnoreSSL:$ignoreSSL)) {
                $allHealthy = $false
            }
        }
        
        if ($allHealthy) {
            Write-ColorOutput "`n[+] All monitoring services are healthy!" "Success"
            Show-AccessUrls
        } else {
            Write-ColorOutput "`n Some services failed health checks" "Warning"
            Write-ColorOutput "Check logs with: .\monitoring.ps1 logs" "Info"
        }
    }
}

# Function to stop monitoring stack
function Stop-Monitoring {
    Write-Header "Stopping Platform Monitoring Stack"
    
    Write-ColorOutput "Stopping monitoring services..." "Info"
    
    $composeCmd = "docker-compose -f monitoring/docker-compose.monitoring.yml down"
    
    if ($RemoveVolumes) {
        $composeCmd += " -v"
        Write-ColorOutput "  Warning: This will remove all monitoring data!" "Warning"
        $confirm = Read-Host "  Are you sure? (y/N)"
        if ($confirm -ne "y") {
            Write-ColorOutput "  Cancelled." "Warning"
            return
        }
    }
    
    Invoke-Expression $composeCmd
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput " Monitoring stack stopped successfully" "Success"
    } else {
        Write-ColorOutput " Failed to stop monitoring stack" "Error"
    }
}

# Function to restart monitoring stack
function Restart-Monitoring {
    Write-Header "Restarting Platform Monitoring Stack"
    Stop-Monitoring
    Start-Sleep -Seconds 2
    Start-Monitoring
}

# Function to show monitoring status
function Show-Status {
    Write-Header "Platform Monitoring Status"
    
    Write-ColorOutput "Container Status:" "Info"
    docker-compose -f monitoring/docker-compose.monitoring.yml ps
    
    Write-ColorOutput "`nResource Usage:" "Info"
    $containers = @(
        "genx-prometheus",
        "genx-grafana",
        "genx-jaeger",
        "genx-loki",
        "genx-alertmanager",
        "genx-otel-collector",
        "genx-cadvisor",
        "genx-node-exporter"
    )
    
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $containers 2>$null
}

# Function to show logs
function Show-Logs {
    param(
        [string]$Service = "",
        [int]$Tail = 100,
        [switch]$Follow = $false
    )
    
    Write-Header "Platform Monitoring Logs"
    
    $composeCmd = "docker-compose -f monitoring/docker-compose.monitoring.yml logs"
    
    if ($Service) {
        $composeCmd += " $Service"
    }
    
    if ($Follow) {
        $composeCmd += " -f"
    }
    
    $composeCmd += " --tail=$Tail"
    
    Invoke-Expression $composeCmd
}

# Function to show access URLs
function Show-AccessUrls {
    Write-Header "Monitoring Access URLs"
    
    $protocol = if ($UseTLS) { "https" } else { "http" }
    
    $urls = @(
        @{Name="Prometheus"; Url="$protocol`://localhost:9090"; Description="Metrics storage and querying"},
        @{Name="Grafana"; Url="$protocol`://localhost:3000"; Description="Dashboards and visualization (admin/changeme)"},
        @{Name="Jaeger"; Url="http://localhost:16686"; Description="Distributed tracing UI"},
        @{Name="AlertManager"; Url="$protocol`://localhost:9093"; Description="Alert management"},
        @{Name="Loki"; Url="http://localhost:3100"; Description="Log aggregation (API only)"}
    )
    
    if ($UseTLS) {
        Write-ColorOutput "TLS is enabled. Services are using HTTPS with self-signed certificates." "Warning"
        Write-ColorOutput "Your browser will show security warnings - this is expected." "Info"
    }
    
    foreach ($item in $urls) {
        Write-ColorOutput "$($item.Name): $($item.Url)" "Info"
        Write-Host "  $($item.Description)"
    }
    
    Write-ColorOutput "`nQuick Commands:" "Info"
    Write-Host "  View Prometheus targets: $protocol`://localhost:9090/targets"
    Write-Host "  Import Grafana dashboard: $protocol`://localhost:3000/dashboard/import"
    Write-Host "  View Jaeger services: http://localhost:16686/search"
    Write-Host "  Check AlertManager status: $protocol`://localhost:9093/#/status"
}

# Function to run monitoring diagnostics
function Start-Diagnostics {
    Write-Header "Platform Monitoring Diagnostics"
    
    Write-ColorOutput "Checking services..." "Info"
    
    # Check containers
    $requiredContainers = @(
        "genx-prometheus",
        "genx-grafana",
        "genx-jaeger",
        "genx-loki",
        "genx-alertmanager",
        "genx-otel-collector"
    )
    
    $runningContainers = docker ps --format "{{.Names}}"
    $missingContainers = @()
    
    foreach ($container in $requiredContainers) {
        if ($runningContainers -contains $container) {
            Write-ColorOutput "   $container is running" "Success"
        } else {
            Write-ColorOutput "   $container is not running" "Error"
            $missingContainers += $container
        }
    }
    
    # Check network connectivity
    Write-ColorOutput "`nChecking network..." "Info"
    $networkInfo = docker network inspect genx-platform-network 2>$null | ConvertFrom-Json
    
    if ($networkInfo) {
        Write-ColorOutput "   Platform network exists" "Success"
        $connectedContainers = $networkInfo.Containers.PSObject.Properties.Name
        Write-Host "  Connected containers: $($connectedContainers.Count)"
    } else {
        Write-ColorOutput "   Platform network not found" "Error"
    }
    
    # Check Prometheus targets
    Write-ColorOutput "`nChecking Prometheus targets..." "Info"
    try {
        $targets = Invoke-RestMethod -Uri "http://localhost:9090/api/v1/targets" -ErrorAction SilentlyContinue
        $upTargets = $targets.data.activeTargets | Where-Object { $_.health -eq "up" }
        $downTargets = $targets.data.activeTargets | Where-Object { $_.health -ne "up" }
        
        Write-ColorOutput "  Active targets: $($targets.data.activeTargets.Count)" "Info"
        Write-ColorOutput "  Healthy targets: $($upTargets.Count)" "Success"
        if ($downTargets.Count -gt 0) {
            Write-ColorOutput "  Unhealthy targets: $($downTargets.Count)" "Warning"
            $downTargets | ForEach-Object {
                Write-Host "    - $($_.labels.job): $($_.lastError)"
            }
        }
    } catch {
        Write-ColorOutput "   Cannot reach Prometheus API" "Error"
    }
    
    # Check disk usage
    Write-ColorOutput "`nChecking disk usage..." "Info"
    $volumes = @("prometheus-data", "grafana-data", "jaeger-data", "loki-data")
    foreach ($volume in $volumes) {
        $volumeInfo = docker volume inspect $volume 2>$null | ConvertFrom-Json
        if ($volumeInfo) {
            Write-Host "  $volume exists at: $($volumeInfo.Mountpoint)"
        }
    }
}

# Function to backup monitoring data
function Backup-MonitoringData {
    param(
        [string]$BackupDir
    )
    
    # Set default backup directory if not provided
    if (-not $BackupDir) {
        $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $BackupDir = Join-Path -Path (Get-Location) -ChildPath "monitoring-backup-$timestamp"
    }
    
    Write-Header "Backing Up Monitoring Data"
    
    Write-ColorOutput "Creating backup directory: $BackupDir" "Info"
    New-Item -ItemType Directory -Path $BackupDir -Force | Out-Null
    
    # Backup Grafana dashboards
    Write-ColorOutput "Backing up Grafana dashboards..." "Info"
    $dashboardsDir = Join-Path $BackupDir "grafana-dashboards"
    New-Item -ItemType Directory -Path $dashboardsDir -Force | Out-Null
    
    if (Test-Path "monitoring/grafana/dashboards") {
        Copy-Item -Path "monitoring/grafana/dashboards/*" -Destination $dashboardsDir -Recurse
        Write-ColorOutput "  [+] Dashboards backed up" "Success"
    }
    
    # Backup Prometheus data (if needed)
    Write-ColorOutput "Note: To backup Prometheus data, use Prometheus snapshot API" "Info"
    Write-Host "  Example: curl -X POST http://localhost:9090/api/v1/admin/tsdb/snapshot"
    
    # Backup configurations
    Write-ColorOutput "`nBacking up configurations..." "Info"
    $configDir = Join-Path $BackupDir "configs"
    New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    
    $configFiles = @(
        "monitoring/prometheus/prometheus.yml",
        "monitoring/grafana/provisioning/datasources/datasources.yml",
        "monitoring/alertmanager/config/config.yml",
        "monitoring/loki/config/loki-config.yaml"
    )
    
    foreach ($file in $configFiles) {
        if (Test-Path $file) {
            $destFile = Join-Path $configDir (Split-Path $file -Leaf)
            Copy-Item -Path $file -Destination $destFile
            Write-ColorOutput "  [+] Backed up $(Split-Path $file -Leaf)" "Success"
        }
    }
    
    Write-ColorOutput "`nBackup completed: $BackupDir" "Success"
}

# Function to show help
function Show-Help {
    Write-Header "$Platform v$Version"
    
    Write-Host "Usage: .\monitoring.ps1 [command] [options]"
    Write-Host ""
    Write-ColorOutput "Commands:" "Info"
    Write-Host "  up, start       Start the monitoring stack"
    Write-Host "  down, stop      Stop the monitoring stack"
    Write-Host "  restart         Restart the monitoring stack"
    Write-Host "  status, ps      Show status of monitoring services"
    Write-Host "  logs            Show logs from monitoring services"
    Write-Host "  urls            Show access URLs for monitoring services"
    Write-Host "  diag            Run diagnostics on monitoring stack"
    Write-Host "  backup          Backup monitoring configurations"
    Write-Host "  help            Show this help message"
    Write-Host ""
    Write-ColorOutput "Options:" "Info"
    Write-Host "  -Environment    Environment name (default: production)"
    Write-Host "  -Detach         Run in detached mode (default: true)"
    Write-Host "  -ForceRecreate  Force recreate containers"
    Write-Host "  -RemoveVolumes  Remove volumes when stopping (WARNING: deletes data)"
    Write-Host "  -UseTLS         Enable TLS/HTTPS for monitoring services"
    Write-Host "  -Timeout        Timeout in seconds (default: 300)"
    Write-Host ""
    Write-ColorOutput "Examples:" "Info"
    Write-Host "  .\monitoring.ps1 up                    # Start monitoring stack"
    Write-Host "  .\monitoring.ps1 up -UseTLS            # Start with TLS enabled"
    Write-Host "  .\monitoring.ps1 logs grafana -Follow  # Follow Grafana logs"
    Write-Host "  .\monitoring.ps1 down -RemoveVolumes   # Stop and remove all data"
    Write-Host "  .\monitoring.ps1 diag                  # Run diagnostics"
    Write-Host ""
    Write-ColorOutput "Monitoring Services:" "Info"
    Write-Host "  - Prometheus:     Metrics storage and querying"
    Write-Host "  - Grafana:        Dashboards and visualization"
    Write-Host "  - Jaeger:         Distributed tracing"
    Write-Host "  - Loki:           Log aggregation"
    Write-Host "  - AlertManager:   Alert routing and management"
    Write-Host "  - OTel Collector: Telemetry collection and routing"
}

# Main execution
switch ($Command.ToLower()) {
    { $_ -in @("up", "start") } {
        Start-Monitoring
    }
    { $_ -in @("down", "stop") } {
        Stop-Monitoring
    }
    "restart" {
        Restart-Monitoring
    }
    { $_ -in @("status", "ps") } {
        Show-Status
    }
    "logs" {
        $logParams = @{}
        if ($args.Count -gt 0) {
            $logParams.Service = $args[0]
        }
        Show-Logs @logParams
    }
    "urls" {
        Show-AccessUrls
    }
    { $_ -in @("diag", "diagnostics") } {
        Start-Diagnostics
    }
    "backup" {
        if ($args.Count -gt 0) {
            Backup-MonitoringData -BackupDir $args[0]
        } else {
            Backup-MonitoringData
        }
    }
    "help" {
        Show-Help
    }
    default {
        Write-ColorOutput "Unknown command: $Command" "Error"
        Write-Host "Run '.\monitoring.ps1 help' for usage information"
        exit 1
    }
}
