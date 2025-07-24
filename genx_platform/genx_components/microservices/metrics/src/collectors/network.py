"""
Network Metrics Collector
Collects network I/O, connection states, and interface statistics
"""
import time
from typing import Dict, Any, List, Optional
import psutil

from .base import BaseCollector
from ..utils.logger import setup_logging

logger = setup_logging(__name__)


class NetworkCollector(BaseCollector):
    """
    Collects network-related metrics including I/O, connections, and interfaces
    """
    
    def __init__(self):
        super().__init__("network")
        self._last_io_counters = None
        self._last_io_time = None
        self._primary_interface = None
        
    async def _initialize(self) -> None:
        """Initialize network collector"""
        # Get initial I/O counters for rate calculation
        try:
            self._last_io_counters = psutil.net_io_counters()
            self._last_io_time = time.time()
        except Exception as e:
            logger.warning(f"Failed to get initial network I/O counters: {e}")
        
        # Try to identify primary network interface
        self._primary_interface = await self._identify_primary_interface()
        
        # Log network interfaces
        interfaces = psutil.net_if_stats()
        logger.info(
            f"Network collector initialized with {len(interfaces)} interfaces",
            primary_interface=self._primary_interface
        )
    
    async def _collect(self) -> Dict[str, Any]:
        """Collect network metrics"""
        metrics = {}
        
        # Network I/O statistics
        with self._time_operation("network_io"):
            metrics['io'] = await self._collect_io_metrics()
        
        # Connection statistics
        with self._time_operation("connections"):
            metrics['connections'] = await self._collect_connection_metrics()
        
        # Network interfaces
        with self._time_operation("interfaces"):
            metrics['interfaces'] = await self._collect_interface_metrics()
        
        # Network health indicators
        metrics['health'] = self._calculate_network_health(metrics)
        
        return metrics
    
    async def _collect_io_metrics(self) -> Dict[str, Any]:
        """Collect network I/O metrics with rate calculation"""
        io_metrics = {}
        
        try:
            current_io = psutil.net_io_counters()
            current_time = time.time()
            
            if current_io:
                io_metrics = {
                    # Cumulative counters
                    'bytes_sent': current_io.bytes_sent,
                    'bytes_recv': current_io.bytes_recv,
                    'packets_sent': current_io.packets_sent,
                    'packets_recv': current_io.packets_recv,
                    'errin': current_io.errin,
                    'errout': current_io.errout,
                    'dropin': current_io.dropin,
                    'dropout': current_io.dropout,
                    
                    # Human-readable cumulative values
                    'gb_sent': self._bytes_to_gb(current_io.bytes_sent),
                    'gb_recv': self._bytes_to_gb(current_io.bytes_recv),
                    
                    # Error rates
                    'error_rate_in': self._safe_divide(current_io.errin, current_io.packets_recv),
                    'error_rate_out': self._safe_divide(current_io.errout, current_io.packets_sent),
                    'drop_rate_in': self._safe_divide(current_io.dropin, current_io.packets_recv),
                    'drop_rate_out': self._safe_divide(current_io.dropout, current_io.packets_sent)
                }
                
                # Calculate rates if we have previous data
                if self._last_io_counters and self._last_io_time:
                    time_delta = current_time - self._last_io_time
                    
                    if time_delta > 0:
                        # Bytes per second
                        bytes_sent_rate = (current_io.bytes_sent - self._last_io_counters.bytes_sent) / time_delta
                        bytes_recv_rate = (current_io.bytes_recv - self._last_io_counters.bytes_recv) / time_delta
                        
                        # Packets per second
                        packets_sent_rate = (current_io.packets_sent - self._last_io_counters.packets_sent) / time_delta
                        packets_recv_rate = (current_io.packets_recv - self._last_io_counters.packets_recv) / time_delta
                        
                        io_metrics['rates'] = {
                            'bytes_sent_per_sec': round(bytes_sent_rate, 2),
                            'bytes_recv_per_sec': round(bytes_recv_rate, 2),
                            'mb_sent_per_sec': round(bytes_sent_rate / (1024 * 1024), 2),
                            'mb_recv_per_sec': round(bytes_recv_rate / (1024 * 1024), 2),
                            'packets_sent_per_sec': round(packets_sent_rate, 2),
                            'packets_recv_per_sec': round(packets_recv_rate, 2),
                            'total_bandwidth_mbps': round((bytes_sent_rate + bytes_recv_rate) * 8 / (1024 * 1024), 2)
                        }
                
                # Update last values
                self._last_io_counters = current_io
                self._last_io_time = current_time
                
        except Exception as e:
            logger.error(f"Failed to collect network I/O metrics: {e}")
            io_metrics = self._get_empty_io_metrics()
        
        return io_metrics
    
    async def _collect_connection_metrics(self) -> Dict[str, Any]:
        """Collect network connection statistics"""
        connection_metrics = {
            'total': 0,
            'by_state': {},
            'by_type': {},
            'by_family': {}
        }
        
        try:
            connections = psutil.net_connections()
            connection_metrics['total'] = len(connections)
            
            # Count connections by state
            for conn in connections:
                # By state (TCP states)
                if conn.status:
                    state = conn.status
                    connection_metrics['by_state'][state] = connection_metrics['by_state'].get(state, 0) + 1
                
                # By type (TCP, UDP)
                conn_type = 'tcp' if conn.type.name == 'SOCK_STREAM' else 'udp'
                connection_metrics['by_type'][conn_type] = connection_metrics['by_type'].get(conn_type, 0) + 1
                
                # By family (IPv4, IPv6)
                family = 'ipv4' if conn.family.name == 'AF_INET' else 'ipv6' if conn.family.name == 'AF_INET6' else 'other'
                connection_metrics['by_family'][family] = connection_metrics['by_family'].get(family, 0) + 1
            
            # Add common connection states summary
            connection_metrics['summary'] = {
                'established': connection_metrics['by_state'].get('ESTABLISHED', 0),
                'listen': connection_metrics['by_state'].get('LISTEN', 0),
                'time_wait': connection_metrics['by_state'].get('TIME_WAIT', 0),
                'close_wait': connection_metrics['by_state'].get('CLOSE_WAIT', 0),
                'active': connection_metrics['by_state'].get('ESTABLISHED', 0) + 
                         connection_metrics['by_state'].get('SYN_SENT', 0) + 
                         connection_metrics['by_state'].get('SYN_RECV', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to collect connection metrics: {e}")
        
        return connection_metrics
    
    async def _collect_interface_metrics(self) -> List[Dict[str, Any]]:
        """Collect metrics for each network interface"""
        interfaces_data = []
        
        try:
            # Get interface addresses and stats
            if_addrs = psutil.net_if_addrs()
            if_stats = psutil.net_if_stats()
            
            # Get per-interface I/O counters
            if_io = psutil.net_io_counters(pernic=True)
            
            for iface_name, addrs in if_addrs.items():
                interface_data = {
                    'name': iface_name,
                    'addresses': []
                }
                
                # Get addresses
                for addr in addrs:
                    addr_info = {
                        'family': addr.family.name,
                        'address': addr.address
                    }
                    if addr.netmask:
                        addr_info['netmask'] = addr.netmask
                    if addr.broadcast:
                        addr_info['broadcast'] = addr.broadcast
                    
                    interface_data['addresses'].append(addr_info)
                
                # Get interface stats
                if iface_name in if_stats:
                    stats = if_stats[iface_name]
                    interface_data.update({
                        'is_up': stats.isup,
                        'speed_mbps': stats.speed,
                        'mtu': stats.mtu,
                        'flags': self._parse_interface_flags(stats)
                    })
                
                # Get interface I/O
                if iface_name in if_io:
                    io = if_io[iface_name]
                    interface_data['io'] = {
                        'bytes_sent': io.bytes_sent,
                        'bytes_recv': io.bytes_recv,
                        'packets_sent': io.packets_sent,
                        'packets_recv': io.packets_recv,
                        'errin': io.errin,
                        'errout': io.errout,
                        'dropin': io.dropin,
                        'dropout': io.dropout,
                        'mb_sent': self._bytes_to_mb(io.bytes_sent),
                        'mb_recv': self._bytes_to_mb(io.bytes_recv)
                    }
                
                # Determine interface type
                interface_data['type'] = self._determine_interface_type(iface_name)
                interface_data['is_primary'] = iface_name == self._primary_interface
                
                interfaces_data.append(interface_data)
                
        except Exception as e:
            logger.error(f"Failed to collect interface metrics: {e}")
        
        # Sort interfaces: primary first, then by name
        interfaces_data.sort(key=lambda x: (not x.get('is_primary', False), x['name']))
        
        return interfaces_data
    
    async def _identify_primary_interface(self) -> Optional[str]:
        """Identify the primary network interface"""
        try:
            # Get default gateway interface
            if_stats = psutil.net_if_stats()
            if_addrs = psutil.net_if_addrs()
            
            # Look for interfaces that are up and have IPv4 addresses
            candidates = []
            for iface, stats in if_stats.items():
                if stats.isup and iface in if_addrs:
                    # Check if has IPv4 address (not loopback)
                    for addr in if_addrs[iface]:
                        if addr.family.name == 'AF_INET' and not addr.address.startswith('127.'):
                            candidates.append((iface, stats.speed))
                            break
            
            # Sort by speed (highest first) and return the fastest
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
            
        except Exception as e:
            logger.warning(f"Failed to identify primary interface: {e}")
        
        return None
    
    def _parse_interface_flags(self, stats) -> List[str]:
        """Parse interface flags into human-readable list"""
        flags = []
        
        if stats.isup:
            flags.append('UP')
        else:
            flags.append('DOWN')
        
        # Additional flags would be parsed from stats.flags if available
        # This is simplified for cross-platform compatibility
        
        return flags
    
    def _determine_interface_type(self, name: str) -> str:
        """Determine the type of network interface"""
        name_lower = name.lower()
        
        if 'lo' in name_lower or 'loopback' in name_lower:
            return 'loopback'
        elif 'eth' in name_lower or 'en' in name_lower:
            return 'ethernet'
        elif 'wlan' in name_lower or 'wifi' in name_lower or 'wl' in name_lower:
            return 'wifi'
        elif 'docker' in name_lower:
            return 'docker'
        elif 'veth' in name_lower:
            return 'virtual_ethernet'
        elif 'tun' in name_lower or 'tap' in name_lower:
            return 'vpn'
        elif 'bridge' in name_lower or 'br' in name_lower:
            return 'bridge'
        else:
            return 'unknown'
    
    def _calculate_network_health(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate network health indicators"""
        health = {
            'status': 'healthy',
            'warnings': [],
            'issues': []
        }
        
        # Check error rates
        if 'io' in metrics:
            error_in = metrics['io'].get('error_rate_in', 0)
            error_out = metrics['io'].get('error_rate_out', 0)
            drop_in = metrics['io'].get('drop_rate_in', 0)
            drop_out = metrics['io'].get('drop_rate_out', 0)
            
            if error_in > 0.01 or error_out > 0.01:  # > 1% errors
                health['warnings'].append(f"High packet error rate: in={error_in:.2%}, out={error_out:.2%}")
                health['status'] = 'degraded'
            
            if drop_in > 0.01 or drop_out > 0.01:  # > 1% drops
                health['warnings'].append(f"High packet drop rate: in={drop_in:.2%}, out={drop_out:.2%}")
                health['status'] = 'degraded'
        
        # Check connection counts
        if 'connections' in metrics:
            total_conns = metrics['connections'].get('total', 0)
            time_wait = metrics['connections'].get('by_state', {}).get('TIME_WAIT', 0)
            close_wait = metrics['connections'].get('by_state', {}).get('CLOSE_WAIT', 0)
            
            if total_conns > 10000:
                health['warnings'].append(f"High number of connections: {total_conns}")
            
            if time_wait > 1000:
                health['warnings'].append(f"High TIME_WAIT connections: {time_wait}")
            
            if close_wait > 100:
                health['warnings'].append(f"High CLOSE_WAIT connections: {close_wait}")
        
        # Check bandwidth usage (if rates available)
        if 'io' in metrics and 'rates' in metrics['io']:
            bandwidth_mbps = metrics['io']['rates'].get('total_bandwidth_mbps', 0)
            if bandwidth_mbps > 900:  # > 900 Mbps on gigabit
                health['warnings'].append(f"High bandwidth usage: {bandwidth_mbps:.1f} Mbps")
        
        # Check interface status
        if 'interfaces' in metrics:
            down_interfaces = [
                iface['name'] for iface in metrics['interfaces'] 
                if not iface.get('is_up', True) and iface.get('type') != 'loopback'
            ]
            if down_interfaces:
                health['issues'].append(f"Interfaces down: {', '.join(down_interfaces)}")
                health['status'] = 'degraded'
        
        return health
    
    def _get_empty_io_metrics(self) -> Dict[str, Any]:
        """Return empty I/O metrics structure"""
        return {
            'bytes_sent': 0,
            'bytes_recv': 0,
            'packets_sent': 0,
            'packets_recv': 0,
            'errin': 0,
            'errout': 0,
            'dropin': 0,
            'dropout': 0,
            'gb_sent': 0,
            'gb_recv': 0,
            'error_rate_in': 0,
            'error_rate_out': 0,
            'drop_rate_in': 0,
            'drop_rate_out': 0
        }
    
    def _get_error_metrics(self, error_message: str) -> Dict[str, Any]:
        """Return network-specific error metrics"""
        return {
            'io': self._get_empty_io_metrics(),
            'connections': {
                'total': 0,
                'by_state': {},
                'by_type': {},
                'by_family': {}
            },
            'interfaces': [],
            'error': True,
            'error_message': error_message,
            '_metadata': {
                'collector': self.name,
                'error': True
            }
        }