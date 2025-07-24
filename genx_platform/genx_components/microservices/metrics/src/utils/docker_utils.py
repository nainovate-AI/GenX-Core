"""
Docker environment utilities
Detects if running in Docker and adjusts paths accordingly
"""
import os
import psutil


def is_docker():
    """Check if we're running inside a Docker container"""
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    
    # Check cgroup
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        return False


def get_host_path(path):
    """Convert path to host path if running in Docker"""
    if is_docker() and path.startswith('/'):
        # Map to /host path
        return f'/host{path}'
    return path


def get_disk_usage_docker():
    """Get disk usage when running in Docker"""
    if is_docker():
        # Use /host/root for actual host disk usage
        return psutil.disk_usage('/host/root')
    else:
        return psutil.disk_usage('/')


def get_process_iter_docker():
    """Get process iterator that works in Docker"""
    if is_docker():
        # When using pid: host, processes are accessible normally
        return psutil.process_iter()
    else:
        return psutil.process_iter()


# Environment configuration for Docker
DOCKER_ENV = {
    'is_docker': is_docker(),
    'proc_path': '/host/proc' if is_docker() else '/proc',
    'sys_path': '/host/sys' if is_docker() else '/sys',
    'root_path': '/host/root' if is_docker() else '/',
}