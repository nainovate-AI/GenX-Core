@echo off
REM genx_platform/genx_components/microservices/metrics/make.bat
REM Windows batch script for metrics service operations

setlocal enabledelayedexpansion

REM Set variables
set PROJECT_ROOT=..\..\..
set SERVICE_NAME=metrics-service
set IMAGE_NAME=genx/%SERVICE_NAME%
set VERSION=1.0.0
set ENVIRONMENT=production

REM Colors for output (Windows 10+)
set GREEN=[92m
set YELLOW=[93m
set RED=[91m
set BLUE=[94m
set NC=[0m

REM Check command
if "%1"=="" goto help
goto %1

:help
echo %GREEN%GenX Metrics Service - Windows Commands%NC%
echo Usage: make.bat [command]
echo.
echo %BLUE%Build Commands:%NC%
echo   %GREEN%build%NC%              - Build production Docker image
echo   %GREEN%build-dev%NC%          - Build development image
echo   %GREEN%security-scan%NC%      - Run security scans on image
echo.
echo %BLUE%Certificate Commands:%NC%
echo   %GREEN%certs-generate%NC%     - Generate TLS certificates
echo   %GREEN%certs-verify%NC%       - Verify certificates
echo.
echo %BLUE%Deployment Commands:%NC%
echo   %GREEN%up%NC%                 - Start production stack
echo   %GREEN%up-full%NC%            - Start with monitoring stack
echo   %GREEN%up-dev%NC%             - Start development stack
echo   %GREEN%down%NC%               - Stop all services
echo.
echo %BLUE%Operations:%NC%
echo   %GREEN%status%NC%             - Check service status
echo   %GREEN%health%NC%             - Health check all services
echo   %GREEN%logs%NC%               - View service logs
echo   %GREEN%test%NC%               - Run tests
echo   %GREEN%clean%NC%              - Clean up everything
goto :eof

:certs-generate
echo %YELLOW%Generating TLS certificates...%NC%
call scripts\generate-certs.bat
goto :eof

:certs-verify
echo %YELLOW%Verifying certificates...%NC%
if exist certs\ca.crt (
    openssl verify -CAfile certs\ca.crt certs\server.crt
    echo %GREEN%Certificates valid%NC%
) else (
    echo %RED%Certificates not found. Run 'make.bat certs-generate' first%NC%
)
goto :eof

:build
echo %YELLOW%Building production Docker image...%NC%
REM Get absolute path to genx_platform (3 levels up)
cd ..\..\..
set GENX_PLATFORM_PATH=%CD%
echo Building from: %GENX_PLATFORM_PATH%

docker build ^
    --build-arg VERSION=%VERSION% ^
    -f genx_components\microservices\metrics\Dockerfile ^
    -t %IMAGE_NAME%:%VERSION% ^
    -t %IMAGE_NAME%:latest ^
    .

if %ERRORLEVEL% EQU 0 (
    echo %GREEN%Build complete: %IMAGE_NAME%:%VERSION%%NC%
) else (
    echo %RED%Build failed with exit code: %ERRORLEVEL%%NC%
)

cd genx_components\microservices\metrics
goto :eof

:build-dev
echo %YELLOW%Building development image...%NC%
cd %PROJECT_ROOT%
docker build ^
    --build-arg VERSION=dev ^
    --target builder ^
    -f genx_components\microservices\metrics\Dockerfile ^
    -t %IMAGE_NAME%:dev ^
    .
cd genx_components\microservices\metrics
echo %GREEN%Dev build complete%NC%
goto :eof

:security-scan
echo %YELLOW%Running security scans...%NC%
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock ^
    aquasec/trivy image %IMAGE_NAME%:%VERSION%
echo %GREEN%Security scan complete%NC%
goto :eof

:up
echo %YELLOW%Starting production stack...%NC%
set VERSION=%VERSION%
set ENVIRONMENT=%ENVIRONMENT%
docker-compose up -d
echo %GREEN%Services started%NC%
call :status
goto :eof

:up-full
echo %YELLOW%Starting full production stack with monitoring...%NC%
set VERSION=%VERSION%
set ENVIRONMENT=%ENVIRONMENT%
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
echo %GREEN%Full stack started%NC%
call :status
goto :eof

:up-dev
echo %YELLOW%Starting development stack...%NC%
set ENVIRONMENT=development
docker-compose up -d
echo %GREEN%Development services started%NC%
call :status
goto :eof

:down
echo %YELLOW%Stopping all services...%NC%
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down
echo %GREEN%Services stopped%NC%
goto :eof

:status
echo %YELLOW%Service Status:%NC%
docker-compose ps
echo.
echo %YELLOW%Resource Usage:%NC%
docker stats --no-stream
goto :eof

:health
echo %YELLOW%Health Checks:%NC%
echo Metrics Service: 
docker exec genx-metrics-service python -m grpc_health.v1.health_check --address=localhost:50056 2>nul && (
    echo %GREEN%Healthy%NC%
) || (
    echo %RED%Unhealthy%NC%
)
goto :eof

:logs
docker-compose logs -f --tail=100 metrics-service
goto :eof

:test
echo %YELLOW%Running tests...%NC%
timeout /t 5 /nobreak > nul
docker run --rm --network metrics_genx-network ^
    -v %CD%\scripts:/scripts ^
    %IMAGE_NAME%:%VERSION% python /scripts/test_production.py
goto :eof

:clean
echo %YELLOW%Cleaning up...%NC%
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml down -v
docker rmi %IMAGE_NAME%:%VERSION% %IMAGE_NAME%:latest 2>nul
rd /s /q certs 2>nul
rd /s /q logs 2>nul
echo %GREEN%Cleanup complete%NC%
goto :eof

:default
echo %RED%Unknown command: %1%NC%
echo Run 'make.bat help' for usage
goto :eof

endlocal
