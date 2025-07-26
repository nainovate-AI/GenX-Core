# genx_platform/genx_components/microservices/metrics/make.ps1
# PowerShell script for metrics service operations

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [string]$Version = "1.0.0",
    [string]$Environment = "production",
    [int]$Replicas = 1
)

# Set variables
$ServiceName = "metrics-service"
$ImageName = "genx/$ServiceName"

# ANSI color codes
$Green = "`e[92m"
$Yellow = "`e[93m"
$Red = "`e[91m"
$Blue = "`e[94m"
$NC = "`e[0m"

# Check if Docker is running
function Test-Docker {
    try {
        docker version | Out-Null
        return $true
    } catch {
        Write-Host "${Red}ERROR: Docker is not running or not installed${NC}"
        Write-Host "Please ensure Docker Desktop is running and try again."
        return $false
    }
}

function Show-Help {
    Write-Host "${Green}GenX Metrics Service - PowerShell Commands${NC}"
    Write-Host "Usage: .\make.ps1 [command] [options]"
    Write-Host ""
    Write-Host "${Blue}Build Commands:${NC}"
    Write-Host "  ${Green}build${NC}              - Build production Docker image"
    Write-Host "  ${Green}build-dev${NC}          - Build development image"
    Write-Host "  ${Green}security-scan${NC}      - Run security scans on image"
    Write-Host ""
    Write-Host "${Blue}Certificate Commands:${NC}"
    Write-Host "  ${Green}certs-generate${NC}     - Generate TLS certificates"
    Write-Host "  ${Green}certs-verify${NC}       - Verify certificates"
    Write-Host ""
    Write-Host "${Blue}Deployment Commands:${NC}"
    Write-Host "  ${Green}up${NC}                 - Start production stack"
    Write-Host "  ${Green}up-full${NC}            - Start with monitoring stack"
    Write-Host "  ${Green}up-simple${NC}          - Start simple stack (no deps)"
    Write-Host "  ${Green}up-dev${NC}             - Start development stack"
    Write-Host "  ${Green}down${NC}               - Stop all services"
    Write-Host ""
    Write-Host "${Blue}Operations:${NC}"
    Write-Host "  ${Green}status${NC}             - Check service status"
    Write-Host "  ${Green}health${NC}             - Health check all services"
    Write-Host "  ${Green}logs${NC}               - View service logs"
    Write-Host "  ${Green}test${NC}               - Run tests"
    Write-Host "  ${Green}clean${NC}              - Clean up everything"
    Write-Host ""
    Write-Host "${Blue}Options:${NC}"
    Write-Host "  -Version            - Set version (default: 1.0.0)"
    Write-Host "  -Environment        - Set environment (default: production)"
    Write-Host "  -Replicas           - Set replicas for scaling"
}

function Invoke-CertsGenerate {
    Write-Host "${Yellow}Generating TLS certificates...${NC}"
    
    # Check if OpenSSL is available
    if (-not (Get-Command openssl -ErrorAction SilentlyContinue)) {
        Write-Host "${Red}ERROR: OpenSSL is not installed or not in PATH${NC}"
        Write-Host "Please install OpenSSL:"
        Write-Host "  Option 1: choco install openssl"
        Write-Host "  Option 2: Download from https://slproweb.com/products/Win32OpenSSL.html"
        return
    }
    
    & ".\scripts\generate-certs.ps1"
}

function Invoke-CertsVerify {
    Write-Host "${Yellow}Verifying certificates...${NC}"
    if (Test-Path "certs\ca.crt") {
        openssl verify -CAfile certs\ca.crt certs\server.crt
        Write-Host "${Green}Certificates valid${NC}"
    } else {
        Write-Host "${Red}Certificates not found. Run '.\make.ps1 certs-generate' first${NC}"
    }
}

function Invoke-Build {
    # Check Docker first
    if (-not (Test-Docker)) {
        return
    }
    
    Write-Host "${Yellow}Building production Docker image...${NC}"
    
    # Get the absolute path to genx_platform (3 levels up)
    $CurrentPath = Get-Location
    $GenxPlatformPath = (Get-Item $CurrentPath).Parent.Parent.Parent.FullName
    
    Write-Host "Building from: $GenxPlatformPath"
    
    # Verify the Dockerfile exists
    $DockerfilePath = Join-Path $CurrentPath "Dockerfile"
    if (-not (Test-Path $DockerfilePath)) {
        Write-Host "${Red}ERROR: Dockerfile not found at $DockerfilePath${NC}"
        return
    }
    
    Push-Location $GenxPlatformPath
    
    try {
        # Verify we're in the right directory
        if (-not (Test-Path "genx_components")) {
            Write-Host "${Red}ERROR: genx_components directory not found. Are you in the right location?${NC}"
            Write-Host "Expected to find genx_components in: $GenxPlatformPath"
            return
        }
        
        docker build `
            --build-arg VERSION=$Version `
            -f "genx_components\microservices\metrics\Dockerfile" `
            -t ${ImageName}:${Version} `
            -t ${ImageName}:latest `
            .
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "${Green}Build complete: ${ImageName}:${Version}${NC}"
        } else {
            Write-Host "${Red}Build failed with exit code: $LASTEXITCODE${NC}"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-BuildDev {
    Write-Host "${Yellow}Building development image...${NC}"
    
    # Get the absolute path to genx_platform (3 levels up)
    $CurrentPath = Get-Location
    $GenxPlatformPath = (Get-Item $CurrentPath).Parent.Parent.Parent.FullName
    
    Write-Host "Building from: $GenxPlatformPath"
    
    Push-Location $GenxPlatformPath
    
    try {
        docker build `
            --build-arg VERSION=dev `
            --target builder `
            -f "genx_components\microservices\metrics\Dockerfile" `
            -t ${ImageName}:dev `
            .
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "${Green}Dev build complete${NC}"
        } else {
            Write-Host "${Red}Build failed with exit code: $LASTEXITCODE${NC}"
        }
    } finally {
        Pop-Location
    }
}

function Invoke-SecurityScan {
    Write-Host "${Yellow}Running security scans...${NC}"
    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock `
        aquasec/trivy image ${ImageName}:${Version}
    Write-Host "${Green}Security scan complete${NC}"
}

function Invoke-UpSimple {
    Write-Host "${Yellow}Starting simple stack (metrics service only)...${NC}"
    $env:VERSION = $Version
    $env:ENVIRONMENT = "development"
    $env:DEBUG = "true"
    docker-compose -f docker-compose.simple.yml up -d
    Write-Host "${Green}Simple stack started${NC}"
    Start-Sleep -Seconds 2
    Invoke-Status
}

function Invoke-Up {
    Write-Host "${Yellow}Starting production stack...${NC}"
    
    # Check if .env exists
    if (-not (Test-Path ".env")) {
        Write-Host "${Red}ERROR: .env file not found!${NC}"
        Write-Host "Run setup.ps1 first or copy .env.template to .env"
        return
    }
    
    $env:VERSION = $Version
    $env:ENVIRONMENT = $Environment
    docker-compose up -d
    Write-Host "${Green}Services started${NC}"
    Start-Sleep -Seconds 2
    Invoke-Status
}

function Invoke-UpFull {
    Write-Host "${Yellow}Starting full production stack with monitoring...${NC}"
    $env:VERSION = $Version
    $env:ENVIRONMENT = $Environment
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
    Write-Host "${Green}Full stack started${NC}"
    Invoke-Status
}

function Invoke-UpDev {
    Write-Host "${Yellow}Starting development stack...${NC}"
    $env:ENVIRONMENT = "development"
    docker-compose up -d
    Write-Host "${Green}Development services started${NC}"
    Invoke-Status
}

function Invoke-Down {
    Write-Host "${Yellow}Stopping all services...${NC}"
    
    # Stop all possible compose configurations
    docker-compose down 2>$null
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down 2>$null
    docker-compose -f docker-compose.simple.yml down 2>$null
    
    Write-Host "${Green}Services stopped${NC}"
    
    # Offer to clean networks
    $cleanNetworks = Read-Host "Clean up networks? (y/N)"
    if ($cleanNetworks -eq 'y') {
        Write-Host "${Yellow}Cleaning up networks...${NC}"
        docker network prune -f
    }
}

function Invoke-Status {
    Write-Host "${Yellow}Service Status:${NC}"
    docker-compose ps
    Write-Host ""
    Write-Host "${Yellow}Resource Usage:${NC}"
    docker stats --no-stream
}

function Invoke-Health {
    Write-Host "${Yellow}Health Checks:${NC}"
    Write-Host -NoNewline "Metrics Service: "
    
    try {
        docker exec genx-metrics-service python -m grpc_health.v1.health_check --address=localhost:50056 2>$null
        Write-Host "${Green}Healthy${NC}"
    } catch {
        Write-Host "${Red}Unhealthy${NC}"
    }
    
    # Check other services if monitoring stack is up
    if (Test-Connection -ComputerName localhost -Port 9090 -Quiet -Count 1) {
        Write-Host -NoNewline "Prometheus: "
        $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            Write-Host "${Green}Healthy${NC}"
        } else {
            Write-Host "${Red}Unhealthy${NC}"
        }
    }
}

function Invoke-Logs {
    docker-compose logs -f --tail=100 metrics-service
}

function Invoke-Test {
    Write-Host "${Yellow}Running tests...${NC}"
    Start-Sleep -Seconds 5
    
    docker run --rm --network metrics_genx-network `
        -v "${PWD}\scripts:/scripts" `
        ${ImageName}:${Version} python /scripts/test_production.py
}

function Invoke-Clean {
    Write-Host "${Yellow}Cleaning up...${NC}"
    docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down -v
    docker rmi ${ImageName}:${Version} ${ImageName}:latest 2>$null
    Remove-Item -Path certs -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -Path logs -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "${Green}Cleanup complete${NC}"
}

function Invoke-Scale {
    Write-Host "${Yellow}Scaling metrics service to $Replicas replicas...${NC}"
    docker-compose up -d --scale metrics-service=$Replicas
    Write-Host "${Green}Scaled to $Replicas replicas${NC}"
}

# Execute the command
switch ($Command) {
    "help" { Show-Help }
    "certs-generate" { Invoke-CertsGenerate }
    "certs-verify" { Invoke-CertsVerify }
    "build" { Invoke-Build }
    "build-dev" { Invoke-BuildDev }
    "security-scan" { Invoke-SecurityScan }
    "up" { Invoke-Up }
    "up-full" { Invoke-UpFull }
    "up-simple" { Invoke-UpSimple }
    "up-dev" { Invoke-UpDev }
    "down" { Invoke-Down }
    "status" { Invoke-Status }
    "health" { Invoke-Health }
    "logs" { Invoke-Logs }
    "test" { Invoke-Test }
    "clean" { Invoke-Clean }
    "scale" { Invoke-Scale }
    default {
        Write-Host "${Red}Unknown command: $Command${NC}"
        Write-Host "Run '.\make.ps1 help' for usage"
    }
}