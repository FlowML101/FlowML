"""
FlowML Cluster Service - Tailscale Mesh Networking Support

Enables distributed worker mesh using Tailscale's WireGuard-based VPN.
Workers across different networks can join the same FlowML cluster.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                 TAILSCALE MESH (100.x.x.x)                  │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
    │  │  Master  │◄──►│ Worker 1 │◄──►│ Worker 2 │              │
    │  │ FastAPI  │    │ Celery   │    │ Celery   │              │
    │  │ Redis    │    │ Worker   │    │ Worker   │              │
    │  └──────────┘    └──────────┘    └──────────┘              │
    │   100.64.0.1      100.64.0.2      100.64.0.3               │
    └─────────────────────────────────────────────────────────────┘

Setup for workers:
    1. Install Tailscale: https://tailscale.com/download
    2. Join tailnet: tailscale up --auth-key=<key>
    3. Start worker: celery -A worker.celery_app worker --broker=redis://master:6379

Benefits:
    - No port forwarding required
    - Works through NAT/firewalls
    - WireGuard encryption (secure)
    - MagicDNS for hostname discovery
    - Auto-reconnect on network changes
"""
import os
import json
import subprocess
import socket
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from loguru import logger


@dataclass
class TailscaleStatus:
    """Tailscale connection status"""
    installed: bool = False
    running: bool = False
    logged_in: bool = False
    tailnet_name: Optional[str] = None
    self_ip: Optional[str] = None
    self_hostname: Optional[str] = None
    dns_suffix: Optional[str] = None
    peers: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.peers is None:
            self.peers = []
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class ClusterNode:
    """A node in the FlowML cluster"""
    hostname: str
    tailscale_ip: Optional[str]
    local_ip: str
    role: str  # "master" or "worker"
    online: bool
    last_seen: str
    capabilities: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class TailscaleService:
    """
    Service for interacting with Tailscale CLI.
    
    Tailscale provides:
    - Mesh VPN: All nodes can communicate directly
    - MagicDNS: hostname.tailnet.ts.net resolution
    - ACLs: Control who can access what
    - HTTPS certs: Free TLS for internal services
    """
    
    def __init__(self):
        self._status_cache: Optional[TailscaleStatus] = None
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 30  # seconds
    
    def _run_tailscale(self, args: List[str], timeout: int = 10) -> tuple[bool, str]:
        """Run tailscale CLI command"""
        try:
            result = subprocess.run(
                ["tailscale"] + args,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip()
        except FileNotFoundError:
            return False, "Tailscale not installed"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)
    
    def is_installed(self) -> bool:
        """Check if Tailscale CLI is installed"""
        success, _ = self._run_tailscale(["version"])
        return success
    
    def get_status(self, use_cache: bool = True) -> TailscaleStatus:
        """
        Get current Tailscale status.
        
        Returns status object with:
        - Connection state
        - Tailnet name
        - Self IP and hostname
        - Peer list
        """
        # Return cached if fresh
        if use_cache and self._status_cache and self._cache_time:
            age = (datetime.utcnow() - self._cache_time).total_seconds()
            if age < self._cache_ttl:
                return self._status_cache
        
        status = TailscaleStatus()
        
        # Check if installed
        success, output = self._run_tailscale(["version"])
        if not success:
            return status
        status.installed = True
        
        # Get detailed status as JSON
        success, output = self._run_tailscale(["status", "--json"])
        if not success:
            return status
        
        try:
            data = json.loads(output)
            
            status.running = True
            status.logged_in = data.get("BackendState") == "Running"
            
            # Self info
            if "Self" in data:
                self_info = data["Self"]
                status.self_hostname = self_info.get("HostName")
                status.self_ip = self_info.get("TailscaleIPs", [None])[0]
                status.dns_suffix = self_info.get("DNSName", "").split(".")[-3] if self_info.get("DNSName") else None
            
            # Tailnet name
            if "CurrentTailnet" in data:
                status.tailnet_name = data["CurrentTailnet"].get("Name")
            
            # Peers
            if "Peer" in data:
                peers = []
                for peer_id, peer_info in data["Peer"].items():
                    peers.append({
                        "id": peer_id,
                        "hostname": peer_info.get("HostName"),
                        "dns_name": peer_info.get("DNSName"),
                        "tailscale_ip": peer_info.get("TailscaleIPs", [None])[0],
                        "online": peer_info.get("Online", False),
                        "os": peer_info.get("OS"),
                        "last_seen": peer_info.get("LastSeen"),
                    })
                status.peers = peers
                
        except json.JSONDecodeError:
            logger.warning("Failed to parse Tailscale status JSON")
        
        # Cache result
        self._status_cache = status
        self._cache_time = datetime.utcnow()
        
        return status
    
    def get_tailscale_ip(self) -> Optional[str]:
        """Get this machine's Tailscale IP (100.x.x.x)"""
        success, output = self._run_tailscale(["ip", "-4"])
        if success:
            return output.split("\n")[0].strip()
        return None
    
    def get_dns_name(self) -> Optional[str]:
        """Get this machine's MagicDNS name (hostname.tailnet.ts.net)"""
        status = self.get_status()
        if status.logged_in and status.self_hostname and status.dns_suffix:
            return f"{status.self_hostname}.{status.dns_suffix}.ts.net"
        return None
    
    def find_peers_with_tag(self, tag: str = "flowml") -> List[Dict[str, Any]]:
        """
        Find peers that might be FlowML workers.
        
        In Tailscale, you can tag devices. Workers should be tagged 'tag:flowml'.
        This requires ACL setup in Tailscale admin.
        """
        status = self.get_status()
        # For now, return all online peers - tag filtering requires ACL setup
        return [p for p in status.peers if p.get("online")]
    
    def ping_peer(self, peer: str, count: int = 1) -> tuple[bool, float]:
        """
        Ping a peer through Tailscale.
        Returns (success, latency_ms).
        """
        success, output = self._run_tailscale(["ping", "--c", str(count), peer])
        if success and "pong from" in output.lower():
            # Parse latency from output like "pong from peer (100.x.x.x) via ... in 15ms"
            try:
                latency_str = output.split(" in ")[-1].replace("ms", "").strip()
                return True, float(latency_str)
            except:
                return True, 0.0
        return False, 0.0


class ClusterService:
    """
    FlowML Cluster Service
    
    Manages the distributed cluster with optional Tailscale support.
    Works in both local and distributed modes.
    
    Modes:
    - Local: Single machine, Redis on localhost
    - LAN: Multiple machines on same network
    - Tailscale: Mesh network across internet
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.tailscale = TailscaleService()
        self._role = os.getenv("FLOWML_ROLE", "auto")  # "master", "worker", or "auto"
    
    @property
    def is_tailscale_enabled(self) -> bool:
        """Check if Tailscale is available and connected"""
        status = self.tailscale.get_status()
        return status.logged_in
    
    @property
    def role(self) -> str:
        """Determine this node's role"""
        if self._role != "auto":
            return self._role
        # Auto-detect: if running FastAPI, likely master
        return os.getenv("FLOWML_ROLE", "master")
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get comprehensive cluster information"""
        ts_status = self.tailscale.get_status()
        
        # Determine Redis connectivity info
        redis_host = self.redis_url.split("://")[1].split(":")[0] if "://" in self.redis_url else "localhost"
        
        return {
            "mode": self._detect_mode(),
            "role": self.role,
            "redis_host": redis_host,
            "redis_url": self.redis_url,
            "tailscale": {
                "enabled": ts_status.logged_in,
                "installed": ts_status.installed,
                "tailnet": ts_status.tailnet_name,
                "self_ip": ts_status.self_ip,
                "self_hostname": ts_status.self_hostname,
                "peer_count": len(ts_status.peers),
                "online_peers": len([p for p in ts_status.peers if p.get("online")]),
            },
            "local": {
                "hostname": socket.gethostname(),
                "local_ip": self._get_local_ip(),
            }
        }
    
    def _detect_mode(self) -> str:
        """Detect cluster mode"""
        ts_status = self.tailscale.get_status()
        if ts_status.logged_in:
            return "tailscale"
        # Check if Redis is remote
        redis_host = self.redis_url.split("://")[1].split(":")[0] if "://" in self.redis_url else "localhost"
        if redis_host not in ("localhost", "127.0.0.1"):
            return "lan"
        return "local"
    
    def _get_local_ip(self) -> str:
        """Get local network IP"""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "127.0.0.1"
    
    def get_worker_join_command(self) -> str:
        """
        Generate the command for workers to join this cluster.
        
        Returns a shell command string that can be run on worker machines.
        """
        ts_status = self.tailscale.get_status()
        
        if ts_status.logged_in and ts_status.self_hostname:
            # Tailscale mode - use MagicDNS hostname
            master_host = ts_status.self_hostname
            if ts_status.dns_suffix:
                master_host = f"{master_host}.{ts_status.dns_suffix}.ts.net"
            return f"celery -A worker.celery_app worker --broker=redis://{master_host}:6379/0 -Q cpu,gpu --hostname=%h-%n"
        else:
            # Local/LAN mode - use IP
            local_ip = self._get_local_ip()
            return f"celery -A worker.celery_app worker --broker=redis://{local_ip}:6379/0 -Q cpu,gpu --hostname=%h-%n"
    
    def get_setup_instructions(self) -> Dict[str, Any]:
        """
        Get cluster setup instructions based on current state.
        
        Returns step-by-step instructions for different scenarios.
        """
        ts_status = self.tailscale.get_status()
        
        instructions = {
            "current_mode": self._detect_mode(),
            "tailscale_status": ts_status.to_dict(),
        }
        
        if not ts_status.installed:
            instructions["setup_tailscale"] = {
                "step": 1,
                "title": "Install Tailscale",
                "description": "Tailscale enables secure mesh networking across different networks.",
                "windows": "winget install tailscale.tailscale",
                "macos": "brew install tailscale",
                "linux": "curl -fsSL https://tailscale.com/install.sh | sh",
                "docs": "https://tailscale.com/download"
            }
        elif not ts_status.logged_in:
            instructions["login_tailscale"] = {
                "step": 2,
                "title": "Login to Tailscale",
                "command": "tailscale up",
                "description": "This will open a browser to authenticate with your Tailscale account."
            }
        else:
            instructions["add_workers"] = {
                "step": 3,
                "title": "Add Worker Nodes",
                "description": f"Run this on any machine with Tailscale connected to join the cluster:",
                "command": self.get_worker_join_command(),
                "prerequisites": [
                    "Python 3.10+ installed",
                    "FlowML backend cloned/installed",
                    f"Tailscale connected to: {ts_status.tailnet_name}"
                ]
            }
        
        return instructions


# Singleton instance
_cluster_service: Optional[ClusterService] = None


def get_cluster_service() -> ClusterService:
    """Get or create cluster service instance"""
    global _cluster_service
    if _cluster_service is None:
        from config import settings
        _cluster_service = ClusterService(redis_url=settings.REDIS_URL)
    return _cluster_service


# Quick probe functions
def get_tailscale_ip() -> Optional[str]:
    """Quick helper to get Tailscale IP"""
    return TailscaleService().get_tailscale_ip()


def is_tailscale_connected() -> bool:
    """Quick helper to check Tailscale status"""
    return TailscaleService().get_status().logged_in
