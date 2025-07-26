@echo off
REM genx_platform/genx_components/microservices/metrics/scripts/generate-certs.bat
REM Generate TLS certificates for production deployment on Windows

setlocal enabledelayedexpansion

REM Configuration
set CERT_DIR=certs
set DAYS_VALID=365
set KEY_SIZE=4096
set COUNTRY=US
set STATE=CA
set CITY=San Francisco
set ORG=GenX Platform
set OU=Engineering

echo Generating TLS certificates for GenX Platform...

REM Create certificate directory
if not exist "%CERT_DIR%" mkdir "%CERT_DIR%"
cd "%CERT_DIR%"

REM Check if OpenSSL is available
where openssl >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: OpenSSL is not installed or not in PATH
    echo Please install OpenSSL for Windows:
    echo   1. Download from https://slproweb.com/products/Win32OpenSSL.html
    echo   2. Install and add to PATH
    echo   3. Or use: choco install openssl
    exit /b 1
)

REM Generate CA private key
echo Generating CA private key...
openssl genrsa -out ca.key %KEY_SIZE%

REM Generate CA certificate
echo Generating CA certificate...
openssl req -new -x509 -days %DAYS_VALID% -key ca.key -out ca.crt ^
    -subj "/C=%COUNTRY%/ST=%STATE%/L=%CITY%/O=%ORG%/OU=%OU%/CN=GenX-CA"

REM Generate server private key
echo Generating server private key...
openssl genrsa -out server.key %KEY_SIZE%

REM Create server certificate config
echo [req] > server.conf
echo distinguished_name = req_distinguished_name >> server.conf
echo req_extensions = v3_req >> server.conf
echo prompt = no >> server.conf
echo. >> server.conf
echo [req_distinguished_name] >> server.conf
echo C = %COUNTRY% >> server.conf
echo ST = %STATE% >> server.conf
echo L = %CITY% >> server.conf
echo O = %ORG% >> server.conf
echo OU = %OU% >> server.conf
echo CN = metrics-service >> server.conf
echo. >> server.conf
echo [v3_req] >> server.conf
echo keyUsage = keyEncipherment, dataEncipherment >> server.conf
echo extendedKeyUsage = serverAuth >> server.conf
echo subjectAltName = @alt_names >> server.conf
echo. >> server.conf
echo [alt_names] >> server.conf
echo DNS.1 = metrics-service >> server.conf
echo DNS.2 = localhost >> server.conf
echo DNS.3 = *.genx.local >> server.conf
echo IP.1 = 127.0.0.1 >> server.conf
echo IP.2 = 172.20.0.0 >> server.conf

REM Generate server certificate request
echo Generating server certificate request...
openssl req -new -key server.key -out server.csr -config server.conf

REM Sign server certificate
echo Signing server certificate...
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial ^
    -out server.crt -days %DAYS_VALID% -extensions v3_req -extfile server.conf

REM Generate client private key
echo Generating client private key...
openssl genrsa -out client.key %KEY_SIZE%

REM Create client certificate config
echo [req] > client.conf
echo distinguished_name = req_distinguished_name >> client.conf
echo req_extensions = v3_req >> client.conf
echo prompt = no >> client.conf
echo. >> client.conf
echo [req_distinguished_name] >> client.conf
echo C = %COUNTRY% >> client.conf
echo ST = %STATE% >> client.conf
echo L = %CITY% >> client.conf
echo O = %ORG% >> client.conf
echo OU = %OU% >> client.conf
echo CN = genx-client >> client.conf
echo. >> client.conf
echo [v3_req] >> client.conf
echo keyUsage = keyEncipherment, dataEncipherment >> client.conf
echo extendedKeyUsage = clientAuth >> client.conf

REM Generate client certificate request
echo Generating client certificate request...
openssl req -new -key client.key -out client.csr -config client.conf

REM Sign client certificate
echo Signing client certificate...
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial ^
    -out client.crt -days %DAYS_VALID% -extensions v3_req -extfile client.conf

REM Clean up
del *.csr *.conf ca.srl 2>nul

REM Verify certificates
echo.
echo Verifying certificates...
openssl verify -CAfile ca.crt server.crt
openssl verify -CAfile ca.crt client.crt

REM Display certificate information
echo.
echo Certificate generation complete!
echo.
echo Certificate information:
echo CA Certificate:
openssl x509 -in ca.crt -text -noout | findstr /C:"Subject:" /C:"Not Before" /C:"Not After"
echo.
echo Server Certificate:
openssl x509 -in server.crt -text -noout | findstr /C:"Subject:" /C:"DNS:" /C:"IP:"
echo.
echo Client Certificate:
openssl x509 -in client.crt -text -noout | findstr /C:"Subject:"

cd ..
echo.
echo TLS certificates generated in %CERT_DIR%\

endlocal