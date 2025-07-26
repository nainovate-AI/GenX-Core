# genx_platform/genx_components/microservices/metrics/scripts/generate-certs.ps1
# Generate TLS certificates for production deployment on Windows

# Configuration
$CertDir = "certs"
$DaysValid = 365
$KeySize = 4096
$Country = "US"
$State = "CA"
$City = "San Francisco"
$Org = "GenX Platform"
$OU = "Engineering"

Write-Host "Generating TLS certificates for GenX Platform..." -ForegroundColor Yellow

# Create certificate directory
if (-not (Test-Path $CertDir)) {
    New-Item -ItemType Directory -Path $CertDir | Out-Null
}

Set-Location $CertDir

try {
    # Generate CA private key
    Write-Host "Generating CA private key..." -ForegroundColor Yellow
    openssl genrsa -out ca.key $KeySize

    # Generate CA certificate
    Write-Host "Generating CA certificate..." -ForegroundColor Yellow
    openssl req -new -x509 -days $DaysValid -key ca.key -out ca.crt `
        -subj "/C=$Country/ST=$State/L=$City/O=$Org/OU=$OU/CN=GenX-CA"

    # Generate server private key
    Write-Host "Generating server private key..." -ForegroundColor Yellow
    openssl genrsa -out server.key $KeySize

    # Create server certificate config
    @"
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $Country
ST = $State
L = $City
O = $Org
OU = $OU
CN = metrics-service

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = serverAuth
subjectAltName = @alt_names

[alt_names]
DNS.1 = metrics-service
DNS.2 = localhost
DNS.3 = *.genx.local
IP.1 = 127.0.0.1
IP.2 = 172.20.0.0
"@ | Out-File -Encoding ASCII server.conf

    # Generate server certificate request
    Write-Host "Generating server certificate request..." -ForegroundColor Yellow
    openssl req -new -key server.key -out server.csr -config server.conf

    # Sign server certificate
    Write-Host "Signing server certificate..." -ForegroundColor Yellow
    openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial `
        -out server.crt -days $DaysValid -extensions v3_req -extfile server.conf

    # Generate client private key
    Write-Host "Generating client private key..." -ForegroundColor Yellow
    openssl genrsa -out client.key $KeySize

    # Create client certificate config
    @"
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $Country
ST = $State
L = $City
O = $Org
OU = $OU
CN = genx-client

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = clientAuth
"@ | Out-File -Encoding ASCII client.conf

    # Generate client certificate request
    Write-Host "Generating client certificate request..." -ForegroundColor Yellow
    openssl req -new -key client.key -out client.csr -config client.conf

    # Sign client certificate
    Write-Host "Signing client certificate..." -ForegroundColor Yellow
    openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial `
        -out client.crt -days $DaysValid -extensions v3_req -extfile client.conf

    # Clean up
    Remove-Item *.csr, *.conf, ca.srl -ErrorAction SilentlyContinue

    # Verify certificates
    Write-Host "`nVerifying certificates..." -ForegroundColor Yellow
    openssl verify -CAfile ca.crt server.crt
    openssl verify -CAfile ca.crt client.crt

    # Display certificate information
    Write-Host "`nCertificate generation complete!" -ForegroundColor Green
    Write-Host "`nCertificate information:" -ForegroundColor Yellow
    
    Write-Host "CA Certificate:" -ForegroundColor Cyan
    openssl x509 -in ca.crt -text -noout | Select-String -Pattern "Subject:|Not Before|Not After"
    
    Write-Host "`nServer Certificate:" -ForegroundColor Cyan
    openssl x509 -in server.crt -text -noout | Select-String -Pattern "Subject:|DNS:|IP:"
    
    Write-Host "`nClient Certificate:" -ForegroundColor Cyan
    openssl x509 -in client.crt -text -noout | Select-String -Pattern "Subject:"

} catch {
    Write-Host "Error generating certificates: $_" -ForegroundColor Red
    exit 1
} finally {
    Set-Location ..
}

Write-Host "`nTLS certificates generated in $CertDir\" -ForegroundColor Green