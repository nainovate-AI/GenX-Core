#!/bin/bash
# Debug helper script

echo "Debug commands available:"
echo "  py-spy top --pid <PID>           # See what's running"
echo "  py-spy record -o trace.svg --pid <PID>  # Record trace"
echo "  py-spy dump --pid <PID>          # Dump thread stacks"
echo ""

# Find service PID
SERVICE_PID=$(pgrep -f 'python.*start_service.py')
if [ -n "$SERVICE_PID" ]; then
    echo "Service PID: $SERVICE_PID"
else
    echo "Service not found running"
fi