# genx_platform/certificates.ps1
# Production-grade certificate management for GenX Platform
# Handles creation, validation, and renewal of TLS certificates for all platform services

param(
    [Parameter(Position=0)]
    [string]$Command = "help",
    
    [string]$Environment = "development",
    [string]$CertDir = "./infrastructure/certs",
    [string]$CACountry = "US",
    [string]$CAState = "California",
    [string]$CALocality = "San Francisco",
    [string]$CAOrganization = "GenX Platform",
    [string]$CAOrganizationalUnit = "Platform Security",
    [string]$CACommonName = "GenX Platform CA",
    [string]$CAEmail = "security@genx.platform",
    [int]$ValidityDays = 365,
    [int]$CAValidityDays = 3650,  # 10 years for CA
    [string[]]$SubjectAltNames = @(),
    [switch]$Force = $false,
    [switch]$SkipValidation = $false
)

# Script configuration
$ErrorActionPreference = "Stop"
$Platform = "GenX Platform Certificate Manager"
$Version = "1.0.0"

# Certificate configuration
$CertConfig = @{
    KeySize = 4096
    SignatureAlgorithm = "sha256"
    
    # Services that need certificates
    Services = @(
        @{
            Name = "platform-gateway"
            CN = "gateway.genx.platform"
            SANs = @("localhost", "gateway", "envoy", "*.genx.platform")
            Purpose = "API Gateway and external access"
        },
        @{
            Name = "monitoring"
            CN = "monitoring.genx.platform"
            SANs = @("localhost", "prometheus", "grafana", "jaeger", "loki", "alertmanager")
            Purpose = "Monitoring stack internal communication"
        },
        @{
            Name = "microservices"
            CN = "services.genx.platform"
            SANs = @("localhost", "*.service.genx.platform", "metrics-service", "llm-service", "embedding-service")
            Purpose = "Microservices internal communication"
        },
        @{
            Name = "consul"
            CN = "consul.genx.platform"
            SANs = @("localhost", "consul", "consul.service.consul", "*.consul")
            Purpose = "Consul service mesh"
        },
        @{
            Name = "otel-collector"
            CN = "otel.genx.platform"
            SANs = @("localhost", "otel-collector", "collector.genx.platform")
            Purpose = "OpenTelemetry collector"
        }
    )
    
    # Client certificates for mTLS
    Clients = @(
        @{
            Name = "admin-client"
            CN = "admin@genx.platform"
            Purpose = "Administrative access"
        },
        @{
            Name = "service-client"
            CN = "service@genx.platform"
            Purpose = "Service-to-service authentication"
        },
        @{
            Name = "monitoring-client"
            CN = "monitoring@genx.platform"
            Purpose = "Monitoring service authentication"
        }
    )
}

# Colors for output
$Colors = @{
    'Success' = 'Green'
    'Error' = 'Red'
    'Warning' = 'Yellow'
    'Info' = 'Cyan'
    'Header' = 'Magenta'
}

# Function to print colored output
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Type = 'Info'
    )
    Write-Host $Message -ForegroundColor $Colors[$Type]
}

# Function to print headers
function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-ColorOutput ("=" * 80) 'Header'
    Write-ColorOutput " $Text" 'Header'
    Write-ColorOutput ("=" * 80) 'Header'
    Write-Host ""
}

# Function to check prerequisites
function Test-Prerequisites {
    Write-ColorOutput "Checking prerequisites..." 'Info'
    
    $prerequisites = @()
    
    # Check OpenSSL
    try {
        $opensslVersion = openssl version 2>$null
        if ($opensslVersion) {
            Write-ColorOutput "  [OK] OpenSSL: $opensslVersion" 'Success'
            $prerequisites += $true
        } else {
            throw "OpenSSL not found"
        }
    } catch {
        Write-ColorOutput "  [ERROR] OpenSSL is not installed or not in PATH" 'Error'
        Write-ColorOutput "    Install OpenSSL or use 'choco install openssl' (Windows)" 'Warning'
        $prerequisites += $false
    }
    
    # Check if running from correct directory
    if (-not (Test-Path "docker-compose.yml") -or -not (Test-Path "genx_components")) {
        Write-ColorOutput "  [ERROR] Not in GenX platform root directory" 'Error'
        $prerequisites += $false
    } else {
        Write-ColorOutput "  [OK] Running from platform root directory" 'Success'
        $prerequisites += $true
    }
    
    # Check certificate directory
    if (-not (Test-Path $CertDir)) {
        Write-ColorOutput "  [INFO] Certificate directory doesn't exist, will create it" 'Warning'
    } else {
        Write-ColorOutput "  [OK] Certificate directory exists" 'Success'
    }
    
    return -not ($prerequisites -contains $false)
}

# Function to create directory structure
function Initialize-CertificateStructure {
    Write-ColorOutput "Initializing certificate directory structure..." 'Info'
    
    $directories = @(
        $CertDir,
        "$CertDir/ca",
        "$CertDir/server",
        "$CertDir/client",
        "$CertDir/tmp",
        "$CertDir/backup"
    )
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-ColorOutput "  [OK] Created $dir" 'Success'
        }
    }
    
    # Create .gitignore to prevent committing certificates
    $gitignorePath = "$CertDir/.gitignore"
    if (-not (Test-Path $gitignorePath)) {
        @"
# Ignore all certificate files
*.crt
*.key
*.csr
*.pem
*.p12
*.pfx
*.jks

# Except example configs
!*.example
!README.md
"@ | Set-Content $gitignorePath
        Write-ColorOutput "  [OK] Created .gitignore for certificate directory" 'Success'
    }
}

# Function to create OpenSSL config
function New-OpenSSLConfig {
    param(
        [string]$ConfigPath,
        [string]$CommonName,
        [string[]]$SubjectAltNames
    )
    
    $sanSection = ""
    if ($SubjectAltNames.Count -gt 0) {
        $sanList = @()
        $dnsIndex = 1
        $ipIndex = 1
        
        foreach ($san in $SubjectAltNames) {
            if ($san -match '^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$') {
                $sanList += "IP.$ipIndex = $san"
                $ipIndex++
            } else {
                $sanList += "DNS.$dnsIndex = $san"
                $dnsIndex++
            }
        }
        
        $sanSection = @"

[alt_names]
$($sanList -join "`n")
"@
    }
    
    $config = @"
[req]
default_bits = $($CertConfig.KeySize)
default_md = $($CertConfig.SignatureAlgorithm)
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $CACountry
ST = $CAState
L = $CALocality
O = $CAOrganization
OU = $CAOrganizationalUnit
CN = $CommonName
emailAddress = $CAEmail

[v3_req]
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names
$sanSection
"@
    
    $config | Set-Content $ConfigPath
}

# Function to create Certificate Authority
function New-CertificateAuthority {
    Write-Header "Creating Certificate Authority"
    
    $caDir = "$CertDir/ca"
    $caKey = "$caDir/ca.key"
    $caCert = "$caDir/ca.crt"
    
    if ((Test-Path $caKey) -and (Test-Path $caCert) -and -not $Force) {
        Write-ColorOutput "Certificate Authority already exists. Use -Force to regenerate." 'Warning'
        return $true
    }
    
    Write-ColorOutput "Generating CA private key..." 'Info'
    $keyCmd = "openssl genpkey -algorithm RSA -out `"$caKey`" -pkeyopt rsa_keygen_bits:$($CertConfig.KeySize)"
    Invoke-Expression $keyCmd
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to generate CA private key" 'Error'
        return $false
    }
    
    Write-ColorOutput "Generating CA certificate..." 'Info'
    # Build the subject string
    $subject = "/C=$CACountry/ST=$CAState/L=$CALocality/O=$CAOrganization/OU=$CAOrganizationalUnit/CN=$CACommonName/emailAddress=$CAEmail"
    
    # For Windows, we need to handle the command differently
    $certCmd = "openssl req -new -x509 -days $CAValidityDays -key `"$caKey`" -out `"$caCert`" -subj `"$subject`""
    
    # Execute the command
    $process = Start-Process -FilePath "openssl" -ArgumentList "req -new -x509 -days $CAValidityDays -key `"$caKey`" -out `"$caCert`" -subj `"$subject`"" -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        # If subj parameter fails, try with config file approach
        Write-ColorOutput "Direct subject specification failed, using config file approach..." 'Warning'
        
        $caConfigPath = "$CertDir/tmp/ca.conf"
        $caConfig = @"
[req]
default_bits = $($CertConfig.KeySize)
default_md = $($CertConfig.SignatureAlgorithm)
distinguished_name = req_distinguished_name
x509_extensions = v3_ca
prompt = no

[req_distinguished_name]
C = $CACountry
ST = $CAState
L = $CALocality
O = $CAOrganization
OU = $CAOrganizationalUnit
CN = $CACommonName
emailAddress = $CAEmail

[v3_ca]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always,issuer
basicConstraints = critical,CA:true
keyUsage = critical, digitalSignature, cRLSign, keyCertSign
"@
        $caConfig | Set-Content $caConfigPath
        
        $certCmd = "openssl req -new -x509 -days $CAValidityDays -key `"$caKey`" -out `"$caCert`" -config `"$caConfigPath`""
        Invoke-Expression $certCmd
        
        # Clean up config file
        Remove-Item $caConfigPath -Force -ErrorAction SilentlyContinue
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "Failed to generate CA certificate" 'Error'
        return $false
    }
    
    # Set appropriate permissions (Windows compatible)
    if ($IsWindows -or $PSVersionTable.Platform -eq 'Win32NT') {
        try {
            $acl = Get-Acl $caKey
            $acl.SetAccessRuleProtection($true, $false)
            $adminRule = New-Object System.Security.AccessControl.FileSystemAccessRule("BUILTIN\Administrators", "FullControl", "Allow")
            $systemRule = New-Object System.Security.AccessControl.FileSystemAccessRule("SYSTEM", "FullControl", "Allow")
            $acl.SetAccessRule($adminRule)
            $acl.SetAccessRule($systemRule)
            Set-Acl $caKey $acl
        } catch {
            Write-ColorOutput "Warning: Could not set strict permissions on CA key" 'Warning'
        }
    }
    
    Write-ColorOutput "[OK] Certificate Authority created successfully" 'Success'
    Write-ColorOutput "  CA Certificate: $caCert" 'Info'
    Write-ColorOutput "  CA Private Key: $caKey" 'Info'
    
    # Display CA certificate info
    Write-ColorOutput "`nCA Certificate Information:" 'Info'
    openssl x509 -in $caCert -noout -subject -issuer -dates
    
    return $true
}

# Function to create server certificate
function New-ServerCertificate {
    param(
        [string]$Name,
        [string]$CommonName,
        [string[]]$SubjectAltNames,
        [string]$Purpose
    )
    
    Write-ColorOutput "`nGenerating certificate for: $Name" 'Info'
    Write-ColorOutput "Purpose: $Purpose" 'Info'
    
    $serverDir = "$CertDir/server"
    $keyPath = "$serverDir/$Name.key"
    $csrPath = "$serverDir/$Name.csr"
    $certPath = "$serverDir/$Name.crt"
    $configPath = "$CertDir/tmp/$Name.conf"
    
    if ((Test-Path $keyPath) -and (Test-Path $certPath) -and -not $Force) {
        Write-ColorOutput "  Certificate already exists. Use -Force to regenerate." 'Warning'
        return $true
    }
    
    # Create OpenSSL config
    New-OpenSSLConfig -ConfigPath $configPath -CommonName $CommonName -SubjectAltNames $SubjectAltNames
    
    # Generate private key
    Write-ColorOutput "  Generating private key..." 'Info'
    $process = Start-Process -FilePath "openssl" -ArgumentList "genpkey -algorithm RSA -out `"$keyPath`" -pkeyopt rsa_keygen_bits:$($CertConfig.KeySize)" -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to generate private key" 'Error'
        return $false
    }
    
    # Generate certificate signing request
    Write-ColorOutput "  Generating certificate signing request..." 'Info'
    $process = Start-Process -FilePath "openssl" -ArgumentList "req -new -key `"$keyPath`" -out `"$csrPath`" -config `"$configPath`"" -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to generate CSR" 'Error'
        return $false
    }
    
    # Sign certificate with CA
    Write-ColorOutput "  Signing certificate with CA..." 'Info'
    $signArgs = @(
        "x509",
        "-req",
        "-in", "`"$csrPath`"",
        "-CA", "`"$CertDir/ca/ca.crt`"",
        "-CAkey", "`"$CertDir/ca/ca.key`"",
        "-CAcreateserial",
        "-out", "`"$certPath`"",
        "-days", "$ValidityDays",
        "-extensions", "v3_req",
        "-extfile", "`"$configPath`""
    )
    
    $process = Start-Process -FilePath "openssl" -ArgumentList $signArgs -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to sign certificate" 'Error'
        return $false
    }
    
    # Create combined PEM file (cert + key)
    $pemPath = "$serverDir/$Name.pem"
    Get-Content $certPath, $keyPath | Set-Content $pemPath
    
    Write-ColorOutput "  [OK] Certificate created successfully" 'Success'
    Write-ColorOutput "    Certificate: $certPath" 'Info'
    Write-ColorOutput "    Private Key: $keyPath" 'Info'
    Write-ColorOutput "    Combined PEM: $pemPath" 'Info'
    
    # Clean up temporary files
    Remove-Item $csrPath -Force -ErrorAction SilentlyContinue
    Remove-Item $configPath -Force -ErrorAction SilentlyContinue
    
    return $true
}

# Function to create client certificate
function New-ClientCertificate {
    param(
        [string]$Name,
        [string]$CommonName,
        [string]$Purpose
    )
    
    Write-ColorOutput "`nGenerating client certificate for: $Name" 'Info'
    Write-ColorOutput "Purpose: $Purpose" 'Info'
    
    $clientDir = "$CertDir/client"
    $keyPath = "$clientDir/$Name.key"
    $csrPath = "$clientDir/$Name.csr"
    $certPath = "$clientDir/$Name.crt"
    $p12Path = "$clientDir/$Name.p12"
    
    if ((Test-Path $keyPath) -and (Test-Path $certPath) -and -not $Force) {
        Write-ColorOutput "  Certificate already exists. Use -Force to regenerate." 'Warning'
        return $true
    }
    
    # Generate private key
    Write-ColorOutput "  Generating private key..." 'Info'
    $process = Start-Process -FilePath "openssl" -ArgumentList "genpkey -algorithm RSA -out `"$keyPath`" -pkeyopt rsa_keygen_bits:$($CertConfig.KeySize)" -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to generate private key" 'Error'
        return $false
    }
    
    # Generate certificate signing request
    Write-ColorOutput "  Generating certificate signing request..." 'Info'
    $clientConfigPath = "$CertDir/tmp/$Name-req.conf"
    $clientConfig = @"
[req]
default_bits = $($CertConfig.KeySize)
default_md = $($CertConfig.SignatureAlgorithm)
distinguished_name = req_distinguished_name
prompt = no

[req_distinguished_name]
C = $CACountry
ST = $CAState
L = $CALocality
O = $CAOrganization
OU = Client
CN = $CommonName
emailAddress = $CAEmail
"@
    $clientConfig | Set-Content $clientConfigPath
    
    $process = Start-Process -FilePath "openssl" -ArgumentList "req -new -key `"$keyPath`" -out `"$csrPath`" -config `"$clientConfigPath`"" -NoNewWindow -Wait -PassThru
    
    Remove-Item $clientConfigPath -Force -ErrorAction SilentlyContinue
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to generate CSR" 'Error'
        return $false
    }
    
    # Create extensions config for client cert
    $extConfig = @"
[client_cert]
basicConstraints = CA:FALSE
keyUsage = critical, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth
"@
    $extConfigPath = "$CertDir/tmp/client-ext.conf"
    $extConfig | Set-Content $extConfigPath
    
    # Sign certificate with CA
    Write-ColorOutput "  Signing certificate with CA..." 'Info'
    $signArgs = @(
        "x509",
        "-req",
        "-in", "`"$csrPath`"",
        "-CA", "`"$CertDir/ca/ca.crt`"",
        "-CAkey", "`"$CertDir/ca/ca.key`"",
        "-CAcreateserial",
        "-out", "`"$certPath`"",
        "-days", "$ValidityDays",
        "-extensions", "client_cert",
        "-extfile", "`"$extConfigPath`""
    )
    
    $process = Start-Process -FilePath "openssl" -ArgumentList $signArgs -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to sign certificate" 'Error'
        return $false
    }
    
    # Create PKCS12 bundle for easy import
    Write-ColorOutput "  Creating PKCS12 bundle..." 'Info'
    $p12Args = @(
        "pkcs12",
        "-export",
        "-out", "`"$p12Path`"",
        "-inkey", "`"$keyPath`"",
        "-in", "`"$certPath`"",
        "-certfile", "`"$CertDir/ca/ca.crt`"",
        "-passout", "pass:changeme"
    )
    
    $process = Start-Process -FilePath "openssl" -ArgumentList $p12Args -NoNewWindow -Wait -PassThru
    
    if ($process.ExitCode -ne 0) {
        Write-ColorOutput "  Failed to create PKCS12 bundle" 'Error'
        return $false
    }
    
    Write-ColorOutput "  [OK] Client certificate created successfully" 'Success'
    Write-ColorOutput "    Certificate: $certPath" 'Info'
    Write-ColorOutput "    Private Key: $keyPath" 'Info'
    Write-ColorOutput "    PKCS12 Bundle: $p12Path" 'Info'
    
    # Clean up temporary files
    Remove-Item $csrPath -Force -ErrorAction SilentlyContinue
    Remove-Item $extConfigPath -Force -ErrorAction SilentlyContinue
    
    return $true
}

# Function to create all certificates
function New-AllCertificates {
    Write-Header "Creating All Platform Certificates"
    
    if (-not (Test-Prerequisites)) {
        Write-ColorOutput "Prerequisites check failed. Exiting." 'Error'
        return $false
    }
    
    Initialize-CertificateStructure
    
    # Create CA first
    if (-not (New-CertificateAuthority)) {
        Write-ColorOutput "Failed to create Certificate Authority" 'Error'
        return $false
    }
    
    # Create server certificates
    Write-Header "Creating Server Certificates"
    
    foreach ($service in $CertConfig.Services) {
        $sans = $service.SANs
        if ($SubjectAltNames.Count -gt 0) {
            $sans += $SubjectAltNames
        }
        
        if (-not (New-ServerCertificate -Name $service.Name -CommonName $service.CN -SubjectAltNames $sans -Purpose $service.Purpose)) {
            Write-ColorOutput "Failed to create certificate for $($service.Name)" 'Error'
            return $false
        }
    }
    
    # Create client certificates
    if ($Environment -eq "production") {
        Write-Header "Creating Client Certificates for mTLS"
        
        foreach ($client in $CertConfig.Clients) {
            if (-not (New-ClientCertificate -Name $client.Name -CommonName $client.CN -Purpose $client.Purpose)) {
                Write-ColorOutput "Failed to create client certificate for $($client.Name)" 'Error'
                return $false
            }
        }
    }
    
    Write-Header "Certificate Generation Complete"
    Write-ColorOutput "[OK] All certificates created successfully" 'Success'
    
    # Create README
    $readmePath = "$CertDir/README.md"
    @"
# GenX Platform Certificates

This directory contains TLS certificates for the GenX platform.

## Certificate Structure

### ca/ - Certificate Authority
- ca.crt - CA certificate (distribute to all services)
- ca.key - CA private key (keep secure!)

### server/ - Server certificates for services
- platform-gateway.* - API Gateway certificates
- monitoring.* - Monitoring stack certificates
- microservices.* - Microservices certificates
- consul.* - Consul service mesh certificates
- otel-collector.* - OpenTelemetry collector certificates

### client/ - Client certificates for mTLS
- admin-client.* - Administrative access
- service-client.* - Service-to-service authentication
- monitoring-client.* - Monitoring authentication

## Usage

### Docker Compose
Mount certificates as volumes:
```yaml
volumes:
  - ./infrastructure/certs/ca/ca.crt:/certs/ca.crt:ro
  - ./infrastructure/certs/server/service.crt:/certs/server.crt:ro
  - ./infrastructure/certs/server/service.key:/certs/server.key:ro
```

### Kubernetes
Create secrets:
```bash
kubectl create secret tls platform-tls \
  --cert=server/platform-gateway.crt \
  --key=server/platform-gateway.key
```

## Certificate Renewal

Certificates are valid for $ValidityDays days. To renew:
```powershell
.\certificates.ps1 renew
```

## Security Notes

1. Keep CA private key secure and backed up
2. Never commit certificates to version control
3. Use appropriate file permissions
4. Rotate certificates before expiry
5. Use mTLS for production environments

Generated on: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Environment: $Environment
"@ | Set-Content $readmePath
    
    return $true
}

# Function to validate certificates
function Test-Certificates {
    Write-Header "Validating Platform Certificates"
    
    $validationPassed = $true
    
    # Check CA
    Write-ColorOutput "Checking Certificate Authority..." 'Info'
    $caCert = "$CertDir/ca/ca.crt"
    
    if (Test-Path $caCert) {
        $certInfo = openssl x509 -in $caCert -noout -dates 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "  [OK] CA certificate is valid" 'Success'
            Write-Host "    $certInfo"
        } else {
            Write-ColorOutput "  [ERROR] CA certificate is invalid" 'Error'
            $validationPassed = $false
        }
    } else {
        Write-ColorOutput "  [ERROR] CA certificate not found" 'Error'
        $validationPassed = $false
    }
    
    # Check server certificates
    Write-ColorOutput "`nChecking Server Certificates..." 'Info'
    
    foreach ($service in $CertConfig.Services) {
        $certPath = "$CertDir/server/$($service.Name).crt"
        
        if (Test-Path $certPath) {
            # Verify certificate
            $verifyCmd = "openssl verify -CAfile `"$caCert`" `"$certPath`" 2>&1"
            $verifyResult = Invoke-Expression $verifyCmd
            
            if ($verifyResult -match "OK") {
                Write-ColorOutput "  [OK] $($service.Name) certificate is valid" 'Success'
                
                # Check expiry
                $expiryInfo = openssl x509 -in $certPath -noout -enddate 2>&1
                $expiryDate = $expiryInfo -replace "notAfter=", ""
                Write-Host "    Expires: $expiryDate"
            } else {
                Write-ColorOutput "  [ERROR] $($service.Name) certificate validation failed" 'Error'
                Write-Host "    $verifyResult"
                $validationPassed = $false
            }
        } else {
            Write-ColorOutput "  [ERROR] $($service.Name) certificate not found" 'Error'
            $validationPassed = $false
        }
    }
    
    return $validationPassed
}

# Function to show certificate information
function Show-CertificateInfo {
    Write-Header "Platform Certificate Information"
    
    # CA Info
    $caCert = "$CertDir/ca/ca.crt"
    if (Test-Path $caCert) {
        Write-ColorOutput "Certificate Authority:" 'Info'
        openssl x509 -in $caCert -noout -subject -issuer -dates -fingerprint -sha256
    }
    
    # Server certificates
    Write-ColorOutput "`nServer Certificates:" 'Info'
    Get-ChildItem "$CertDir/server/*.crt" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-ColorOutput "`n$($_.Name):" 'Info'
        openssl x509 -in $_.FullName -noout -subject -dates -ext subjectAltName
    }
    
    # Show usage instructions
    Write-Header "Certificate Usage Examples"
    
    Write-ColorOutput "Docker Compose:" 'Info'
    Write-Host @"
    volumes:
      - ./infrastructure/certs/ca/ca.crt:/certs/ca.crt:ro
      - ./infrastructure/certs/server/monitoring.crt:/certs/server.crt:ro
      - ./infrastructure/certs/server/monitoring.key:/certs/server.key:ro
"@
    
    Write-ColorOutput "`nEnvironment Variables:" 'Info'
    Write-Host @"
    environment:
      - GRPC_TLS_CERT=/certs/server.crt
      - GRPC_TLS_KEY=/certs/server.key
      - GRPC_TLS_CA=/certs/ca.crt
"@
}

# Function to backup certificates
function Backup-Certificates {
    param(
        [string]$BackupPath = "./certificate-backup-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
    )
    
    Write-Header "Backing Up Certificates"
    
    if (-not (Test-Path $CertDir)) {
        Write-ColorOutput "No certificates found to backup" 'Warning'
        return
    }
    
    Write-ColorOutput "Creating backup at: $BackupPath" 'Info'
    
    # Create backup directory
    New-Item -ItemType Directory -Path $BackupPath -Force | Out-Null
    
    # Copy certificates
    Copy-Item -Path $CertDir -Destination $BackupPath -Recurse -Force
    
    # Create backup info
    @{
        BackupDate = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        Environment = $Environment
        CertificateCount = (Get-ChildItem "$CertDir/**/*.crt" -Recurse).Count
        Platform = $Platform
        Version = $Version
    } | ConvertTo-Json | Set-Content "$BackupPath/backup-info.json"
    
    # Create tar archive
    $tarPath = "$BackupPath.tar.gz"
    tar -czf $tarPath -C (Split-Path $BackupPath -Parent) (Split-Path $BackupPath -Leaf)
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "[OK] Backup created successfully: $tarPath" 'Success'
        Remove-Item $BackupPath -Recurse -Force
    } else {
        Write-ColorOutput "Failed to create backup archive" 'Error'
    }
}

# Function to import CA certificate to system store (Windows)
function Import-CACertificate {
    Write-Header "Importing CA Certificate to System Store"
    
    if ($IsWindows -or $PSVersionTable.Platform -eq 'Win32NT') {
        $caCert = "$CertDir/ca/ca.crt"
        
        if (-not (Test-Path $caCert)) {
            Write-ColorOutput "CA certificate not found" 'Error'
            return
        }
        
        Write-ColorOutput "Importing CA certificate to Windows certificate store..." 'Info'
        Write-ColorOutput "This requires administrator privileges" 'Warning'
        
        try {
            $cert = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2($caCert)
            $store = New-Object System.Security.Cryptography.X509Certificates.X509Store("Root", "LocalMachine")
            $store.Open("ReadWrite")
            $store.Add($cert)
            $store.Close()
            
            Write-ColorOutput "[OK] CA certificate imported successfully" 'Success'
        } catch {
            Write-ColorOutput "Failed to import CA certificate: $_" 'Error'
        }
    } else {
        Write-ColorOutput "This command is only supported on Windows" 'Warning'
        Write-ColorOutput "On Linux/Mac, add the CA certificate manually to your system store" 'Info'
        Write-ColorOutput "CA Certificate location: $CertDir/ca/ca.crt" 'Info'
    }
}

# Function to show help
function Show-Help {
    Write-Header "$Platform v$Version"
    
    Write-Host "Usage: .\certificates.ps1 [command] [options]"
    Write-Host ""
    Write-ColorOutput "Commands:" 'Info'
    Write-Host "  create          Create all platform certificates"
    Write-Host "  validate        Validate existing certificates"
    Write-Host "  info            Show certificate information"
    Write-Host "  backup          Backup certificates"
    Write-Host "  import-ca       Import CA cert to system store (Windows)"
    Write-Host "  clean           Remove all certificates (dangerous!)"
    Write-Host "  help            Show this help message"
    Write-Host ""
    Write-ColorOutput "Options:" 'Info'
    Write-Host "  -Environment    Environment name (development/production)"
    Write-Host "  -CertDir        Certificate directory (default: ./infrastructure/certs)"
    Write-Host "  -ValidityDays   Certificate validity in days (default: 365)"
    Write-Host "  -Force          Force regeneration of existing certificates"
    Write-Host "  -SkipValidation Skip certificate validation after creation"
    Write-Host ""
    Write-ColorOutput "Certificate Types Created:" 'Info'
    Write-Host "  - Certificate Authority (self-signed)"
    Write-Host "  - Server certificates for all platform services"
    Write-Host "  - Client certificates for mTLS (production only)"
    Write-Host ""
    Write-ColorOutput "Examples:" 'Info'
    Write-Host "  .\certificates.ps1 create                    # Create development certificates"
    Write-Host "  .\certificates.ps1 create -Environment production -Force"
    Write-Host "  .\certificates.ps1 validate                  # Validate certificates"
    Write-Host "  .\certificates.ps1 backup                    # Backup certificates"
    Write-Host ""
    Write-ColorOutput "Security Notes:" 'Info'
    Write-Host "  - Keep CA private key secure"
    Write-Host "  - Never commit certificates to version control"
    Write-Host "  - Rotate certificates before expiry"
    Write-Host "  - Use mTLS for production environments"
}

# Function to clean certificates
function Remove-AllCertificates {
    Write-Header "Remove All Certificates"
    
    Write-ColorOutput "WARNING: This will delete all certificates!" 'Warning'
    Write-ColorOutput "This action cannot be undone." 'Warning'
    
    $confirm = Read-Host "`nAre you sure you want to delete all certificates? Type 'DELETE' to confirm"
    
    if ($confirm -eq 'DELETE') {
        Write-ColorOutput "Backing up before deletion..." 'Info'
        Backup-Certificates
        
        Write-ColorOutput "Removing certificates..." 'Info'
        Remove-Item -Path $CertDir -Recurse -Force -ErrorAction SilentlyContinue
        
        Write-ColorOutput "[OK] All certificates removed" 'Success'
    } else {
        Write-ColorOutput "Deletion cancelled" 'Info'
    }
}

# Main execution
switch ($Command.ToLower()) {
    'create' {
        New-AllCertificates
        if (-not $SkipValidation) {
            Test-Certificates | Out-Null
        }
    }
    'validate' {
        if (Test-Certificates) {
            Write-ColorOutput "`n[OK] All certificates are valid" 'Success'
        } else {
            Write-ColorOutput "`n[ERROR] Certificate validation failed" 'Error'
            exit 1
        }
    }
    'info' {
        Show-CertificateInfo
    }
    'backup' {
        if ($args.Count -gt 0) {
            Backup-Certificates -BackupPath $args[0]
        } else {
            Backup-Certificates
        }
    }
    'import-ca' {
        Import-CACertificate
    }
    'clean' {
        Remove-AllCertificates
    }
    'help' {
        Show-Help
    }
    default {
        Write-ColorOutput "Unknown command: $Command" 'Error'
        Write-Host "Run '.\certificates.ps1 help' for usage information"
        exit 1
    }
}