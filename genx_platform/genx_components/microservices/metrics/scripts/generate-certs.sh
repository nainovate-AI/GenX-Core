#!/bin/bash
# genx_platform/genx_components/microservices/metrics/scripts/generate-certs.sh
# Generate TLS certificates for production deployment

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
CERT_DIR="./certs"
DAYS_VALID=365
KEY_SIZE=4096
COUNTRY="US"
STATE="CA"
CITY="San Francisco"
ORG="GenX Platform"
OU="Engineering"

echo -e "${YELLOW}Generating TLS certificates for GenX Platform...${NC}"

# Create certificate directory
mkdir -p "$CERT_DIR"
cd "$CERT_DIR"

# Generate CA private key
echo -e "${YELLOW}Generating CA private key...${NC}"
openssl genrsa -out ca.key $KEY_SIZE

# Generate CA certificate
echo -e "${YELLOW}Generating CA certificate...${NC}"
openssl req -new -x509 -days $DAYS_VALID -key ca.key -out ca.crt \
    -subj "/C=$COUNTRY/ST=$STATE/L=$CITY/O=$ORG/OU=$OU/CN=GenX-CA"

# Generate server private key
echo -e "${YELLOW}Generating server private key...${NC}"
openssl genrsa -out server.key $KEY_SIZE

# Create server certificate config
cat > server.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $COUNTRY
ST = $STATE
L = $CITY
O = $ORG
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
EOF

# Generate server certificate request
echo -e "${YELLOW}Generating server certificate request...${NC}"
openssl req -new -key server.key -out server.csr -config server.conf

# Sign server certificate
echo -e "${YELLOW}Signing server certificate...${NC}"
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -days $DAYS_VALID -extensions v3_req -extfile server.conf

# Generate client private key
echo -e "${YELLOW}Generating client private key...${NC}"
openssl genrsa -out client.key $KEY_SIZE

# Create client certificate config
cat > client.conf <<EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req
prompt = no

[req_distinguished_name]
C = $COUNTRY
ST = $STATE
L = $CITY
O = $ORG
OU = $OU
CN = genx-client

[v3_req]
keyUsage = keyEncipherment, dataEncipherment
extendedKeyUsage = clientAuth
EOF

# Generate client certificate request
echo -e "${YELLOW}Generating client certificate request...${NC}"
openssl req -new -key client.key -out client.csr -config client.conf

# Sign client certificate
echo -e "${YELLOW}Signing client certificate...${NC}"
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out client.crt -days $DAYS_VALID -extensions v3_req -extfile client.conf

# Set appropriate permissions
chmod 600 *.key
chmod 644 *.crt

# Clean up
rm -f *.csr *.conf ca.srl

# Verify certificates
echo -e "${YELLOW}Verifying certificates...${NC}"
openssl verify -CAfile ca.crt server.crt
openssl verify -CAfile ca.crt client.crt

# Display certificate information
echo -e "${GREEN}Certificate generation complete!${NC}"
echo -e "${YELLOW}Certificate information:${NC}"
echo "CA Certificate:"
openssl x509 -in ca.crt -text -noout | grep -E "(Subject:|Validity|Not)"
echo ""
echo "Server Certificate:"
openssl x509 -in server.crt -text -noout | grep -E "(Subject:|DNS:|IP:|Validity|Not)"
echo ""
echo "Client Certificate:"
openssl x509 -in client.crt -text -noout | grep -E "(Subject:|Validity|Not)"

cd ..
echo -e "${GREEN}✓ TLS certificates generated in $CERT_DIR${NC}"