# genx_platform/genx_components/microservices/metrics/scripts/check-ports.ps1
# Script to check which ports are in use

Write-Host "Checking ports used by services..." -ForegroundColor Green
Write-Host "==================================" -ForegroundColor Green

$ports = @(
    @{Port=50056; Service="Metrics Service (gRPC)"},
    @{Port=9090; Service="Prometheus"},
    @{Port=9091; Service="Metrics Service (Prometheus)"},
    @{Port=3001; Service="Grafana"},
    @{Port=16686; Service="Jaeger UI"},
    @{Port=8500; Service="Consul"},
    @{Port=9093; Service="AlertManager"},
    @{Port=3100; Service="Loki"},
    @{Port=8080; Service="cAdvisor"},
    @{Port=4317; Service="OTLP gRPC"},
    @{Port=4318; Service="OTLP HTTP"},
    @{Port=8888; Service="OTel Collector Metrics"},
    @{Port=8889; Service="OTel Collector Prometheus"},
    @{Port=9100; Service="Node Exporter"},
    @{Port=9901; Service="Envoy Admin"},
    @{Port=50050; Service="Envoy gRPC"}
)

Write-Host "`nPort Status:" -ForegroundColor Yellow
foreach ($port in $ports) {
    Write-Host -NoNewline "$($port.Port) ($($port.Service)): " -ForegroundColor Cyan
    
    $tcpConnection = Get-NetTCPConnection -LocalPort $port.Port -ErrorAction SilentlyContinue
    if ($tcpConnection) {
        Write-Host "IN USE" -ForegroundColor Red
        $process = Get-Process -Id $tcpConnection[0].OwningProcess -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "  Process: $($process.Name) (PID: $($process.Id))" -ForegroundColor Gray
        }
    } else {
        Write-Host "Available" -ForegroundColor Green
    }
}

Write-Host "`nDocker containers using ports:" -ForegroundColor Yellow
docker ps --format "table {{.Names}}\t{{.Ports}}" | Select-String -Pattern "0.0.0.0"