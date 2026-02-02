"""
Cluster routes - Tailscale mesh networking and distributed cluster management

Provides:
- Cluster status and mode detection
- Tailscale integration status
- Worker join instructions
- Network topology visualization data
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from loguru import logger

from services.cluster import get_cluster_service, TailscaleStatus

router = APIRouter(prefix="/cluster", tags=["cluster"])


class ClusterInfo(BaseModel):
    """Cluster information response"""
    mode: str  # "local", "lan", or "tailscale"
    role: str  # "master" or "worker"
    redis_host: str
    redis_url: str
    tailscale: Dict[str, Any]
    local: Dict[str, Any]


class JoinInstructions(BaseModel):
    """Worker join instructions"""
    command: str
    prerequisites: List[str]
    mode: str
    master_address: str


class SetupInstructions(BaseModel):
    """Full setup instructions"""
    current_mode: str
    tailscale_status: Dict[str, Any]
    setup_tailscale: Optional[Dict[str, Any]] = None
    login_tailscale: Optional[Dict[str, Any]] = None
    add_workers: Optional[Dict[str, Any]] = None


class NetworkTopology(BaseModel):
    """Network topology for visualization"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


@router.get("/info", response_model=ClusterInfo)
async def get_cluster_info():
    """
    Get comprehensive cluster information.
    
    Returns:
    - Current mode (local/lan/tailscale)
    - This node's role (master/worker)
    - Redis connection info
    - Tailscale status if available
    """
    service = get_cluster_service()
    info = service.get_cluster_info()
    return ClusterInfo(**info)


@router.get("/join", response_model=JoinInstructions)
async def get_join_instructions():
    """
    Get the command to join this cluster as a worker.
    
    Workers need to:
    1. Have Python + FlowML installed
    2. Be on the same Tailscale network (if using Tailscale)
    3. Run the provided Celery command
    """
    service = get_cluster_service()
    info = service.get_cluster_info()
    
    return JoinInstructions(
        command=service.get_worker_join_command(),
        prerequisites=[
            "Python 3.10+ installed",
            "FlowML backend installed (pip install -e .)",
            "Redis accessible from worker",
            f"Same network/Tailscale tailnet as master"
        ],
        mode=info["mode"],
        master_address=info["tailscale"]["self_ip"] or info["local"]["local_ip"]
    )


@router.get("/setup", response_model=SetupInstructions)
async def get_setup_instructions():
    """
    Get step-by-step cluster setup instructions.
    
    Returns context-aware instructions based on:
    - Whether Tailscale is installed
    - Whether logged in
    - Current cluster state
    """
    service = get_cluster_service()
    instructions = service.get_setup_instructions()
    return SetupInstructions(**instructions)


@router.get("/tailscale/status")
async def get_tailscale_status():
    """
    Get detailed Tailscale status.
    
    Returns:
    - Installation status
    - Connection status
    - Tailnet name
    - Self IP and hostname
    - Peer list
    """
    service = get_cluster_service()
    status = service.tailscale.get_status(use_cache=False)
    return status.to_dict()


@router.get("/tailscale/peers")
async def get_tailscale_peers():
    """
    Get list of Tailscale peers.
    
    Returns all devices on the same Tailscale network
    that could potentially be FlowML workers.
    """
    service = get_cluster_service()
    status = service.tailscale.get_status()
    
    if not status.logged_in:
        return {
            "enabled": False,
            "message": "Tailscale not connected",
            "peers": []
        }
    
    return {
        "enabled": True,
        "tailnet": status.tailnet_name,
        "self": {
            "hostname": status.self_hostname,
            "ip": status.self_ip,
        },
        "peers": status.peers
    }


@router.post("/tailscale/ping/{hostname}")
async def ping_tailscale_peer(hostname: str):
    """
    Ping a Tailscale peer to check connectivity.
    
    Returns latency in milliseconds.
    """
    service = get_cluster_service()
    success, latency = service.tailscale.ping_peer(hostname)
    
    return {
        "success": success,
        "hostname": hostname,
        "latency_ms": latency if success else None
    }


@router.get("/topology", response_model=NetworkTopology)
async def get_network_topology():
    """
    Get network topology for visualization.
    
    Returns nodes and edges for drawing a cluster diagram.
    This can be used by the frontend for a visual cluster map.
    """
    service = get_cluster_service()
    info = service.get_cluster_info()
    ts_status = service.tailscale.get_status()
    
    nodes = []
    edges = []
    
    # Add self (master) node
    master_node = {
        "id": "master",
        "type": "master",
        "hostname": info["local"]["hostname"],
        "ip": ts_status.self_ip or info["local"]["local_ip"],
        "online": True,
        "services": ["fastapi", "redis", "scheduler"]
    }
    nodes.append(master_node)
    
    # Add Tailscale peers as potential workers
    if ts_status.logged_in:
        for peer in ts_status.peers:
            node = {
                "id": peer.get("id", peer.get("hostname")),
                "type": "peer",
                "hostname": peer.get("hostname"),
                "ip": peer.get("tailscale_ip"),
                "online": peer.get("online", False),
                "os": peer.get("os"),
                "services": []  # Unknown until they register
            }
            nodes.append(node)
            
            # Add edge from master to peer
            if peer.get("online"):
                edges.append({
                    "from": "master",
                    "to": peer.get("id", peer.get("hostname")),
                    "type": "tailscale"
                })
    
    return NetworkTopology(
        nodes=nodes,
        edges=edges,
        stats={
            "total_nodes": len(nodes),
            "online_nodes": len([n for n in nodes if n["online"]]),
            "mode": info["mode"],
            "tailnet": ts_status.tailnet_name
        }
    )


@router.get("/health")
async def cluster_health_check():
    """
    Quick cluster health check.
    
    Checks:
    - Redis connectivity
    - Tailscale status (if enabled)
    - Basic cluster readiness
    """
    import aioredis
    from config import settings
    
    health = {
        "status": "healthy",
        "checks": {}
    }
    
    # Check Redis
    try:
        redis = await aioredis.from_url(settings.REDIS_URL)
        await redis.ping()
        await redis.close()
        health["checks"]["redis"] = {"status": "ok", "url": settings.REDIS_URL}
    except Exception as e:
        health["status"] = "degraded"
        health["checks"]["redis"] = {"status": "error", "message": str(e)}
    
    # Check Tailscale
    service = get_cluster_service()
    ts_status = service.tailscale.get_status()
    health["checks"]["tailscale"] = {
        "status": "ok" if ts_status.logged_in else "not_connected",
        "installed": ts_status.installed,
        "connected": ts_status.logged_in,
        "tailnet": ts_status.tailnet_name
    }
    
    return health
