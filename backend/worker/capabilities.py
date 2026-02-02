"""
Worker Capability Probing
Detects hardware (CPU, RAM, GPU, VRAM), software (Python, ML libs), and network (Tailscale)
"""
import os
import sys
import platform
import socket
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from datetime import datetime
import json

import psutil
from loguru import logger


@dataclass
class GPUInfo:
    """GPU hardware information"""
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: Optional[str] = None
    driver_version: Optional[str] = None


@dataclass
class TailscaleInfo:
    """Tailscale network information"""
    installed: bool = False
    connected: bool = False
    tailscale_ip: Optional[str] = None
    hostname: Optional[str] = None
    tailnet: Optional[str] = None


@dataclass
class WorkerCapabilities:
    """Complete worker capability report"""
    # Identity
    worker_id: str
    hostname: str
    ip_address: str
    
    # Hardware - CPU/RAM
    cpu_count: int
    cpu_count_logical: int
    cpu_freq_mhz: Optional[float]
    total_ram_gb: float
    available_ram_gb: float
    
    # Hardware - GPU
    has_gpu: bool = False
    gpu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    total_vram_gb: float = 0.0
    cuda_available: bool = False
    
    # Network - Tailscale
    tailscale: Optional[TailscaleInfo] = None
    
    # Software
    python_version: str = ""
    os_name: str = ""
    os_version: str = ""
    
    # ML Libraries
    torch_version: Optional[str] = None
    torch_cuda: bool = False
    sklearn_version: Optional[str] = None
    xgboost_version: Optional[str] = None
    lightgbm_version: Optional[str] = None
    pycaret_version: Optional[str] = None
    optuna_version: Optional[str] = None
    
    # Runtime config
    max_concurrency: int = 1
    tags: List[str] = field(default_factory=list)
    
    # Timestamps
    probed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        # Convert GPUInfo objects
        data["gpus"] = [asdict(gpu) if isinstance(gpu, GPUInfo) else gpu for gpu in self.gpus]
        # Convert TailscaleInfo if present
        if self.tailscale:
            data["tailscale"] = asdict(self.tailscale) if isinstance(self.tailscale, TailscaleInfo) else self.tailscale
        return data
    
    def get_best_ip(self) -> str:
        """Get the best IP address for this worker (Tailscale preferred)"""
        if self.tailscale and self.tailscale.connected and self.tailscale.tailscale_ip:
            return self.tailscale.tailscale_ip
        return self.ip_address
    
    def get_queues(self) -> List[str]:
        """Determine which task queues this worker should subscribe to"""
        queues = ["cpu"]  # All workers can do CPU tasks
        
        if self.has_gpu and self.total_vram_gb >= 4:
            queues.append("gpu")
            
            if self.total_vram_gb >= 6:
                queues.append("gpu.vram6")
            if self.total_vram_gb >= 8:
                queues.append("gpu.vram8")
            if self.total_vram_gb >= 12:
                queues.append("gpu.vram12")
            if self.total_vram_gb >= 24:
                queues.append("gpu.vram24")
        
        # LLM capability (needs Ollama check separately)
        if self.has_gpu and self.total_vram_gb >= 6:
            queues.append("llm")
        
        return queues


def _get_gpu_info() -> tuple[bool, int, List[GPUInfo], float, bool, Optional[str]]:
    """Probe NVIDIA GPU using pynvml"""
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return False, 0, [], 0.0, False, None
        
        gpus = []
        total_vram = 0.0
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        
        # Try to get CUDA version
        cuda_version = None
        try:
            cuda_version = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            cuda_version = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
        except:
            pass
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_total = mem_info.total / (1024**3)
            vram_free = mem_info.free / (1024**3)
            total_vram += vram_total
            
            gpus.append(GPUInfo(
                index=i,
                name=name,
                vram_total_gb=round(vram_total, 2),
                vram_free_gb=round(vram_free, 2),
                cuda_version=cuda_version,
                driver_version=driver_version
            ))
        
        pynvml.nvmlShutdown()
        return True, device_count, gpus, round(total_vram, 2), True, cuda_version
        
    except ImportError:
        logger.debug("pynvml not installed, skipping NVIDIA GPU detection")
        return False, 0, [], 0.0, False, None
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
        return False, 0, [], 0.0, False, None


def _get_torch_info() -> tuple[Optional[str], bool]:
    """Check PyTorch installation and CUDA support"""
    try:
        import torch
        version = torch.__version__
        cuda = torch.cuda.is_available()
        return version, cuda
    except ImportError:
        return None, False


def _get_lib_version(module_name: str) -> Optional[str]:
    """Safely get a library version"""
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return None


def _get_ip_address() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def _get_tailscale_info() -> TailscaleInfo:
    """Probe Tailscale installation and connection status"""
    info = TailscaleInfo()
    
    try:
        # Check if tailscale CLI is available
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return info
        
        info.installed = True
        data = json.loads(result.stdout)
        
        # Check if connected
        info.connected = data.get("BackendState") == "Running"
        
        if "Self" in data:
            self_info = data["Self"]
            info.hostname = self_info.get("HostName")
            ips = self_info.get("TailscaleIPs", [])
            if ips:
                info.tailscale_ip = ips[0]  # IPv4
        
        if "CurrentTailnet" in data:
            info.tailnet = data["CurrentTailnet"].get("Name")
            
    except FileNotFoundError:
        logger.debug("Tailscale not installed")
    except subprocess.TimeoutExpired:
        logger.debug("Tailscale status timeout")
    except json.JSONDecodeError:
        logger.debug("Failed to parse Tailscale status")
    except Exception as e:
        logger.debug(f"Tailscale probe failed: {e}")
    
    return info


def probe_capabilities(worker_id: Optional[str] = None) -> WorkerCapabilities:
    """
    Probe all hardware and software capabilities of this machine.
    Returns a WorkerCapabilities object ready for registration.
    """
    logger.info("Probing worker capabilities...")
    
    # Generate worker ID if not provided
    if not worker_id:
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
    
    # CPU info
    cpu_count = psutil.cpu_count(logical=False) or 1
    cpu_count_logical = psutil.cpu_count(logical=True) or 1
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.current if cpu_freq else None
    except:
        cpu_freq_mhz = None
    
    # RAM info
    mem = psutil.virtual_memory()
    total_ram_gb = round(mem.total / (1024**3), 2)
    available_ram_gb = round(mem.available / (1024**3), 2)
    
    # GPU info
    has_gpu, gpu_count, gpus, total_vram, cuda_available, cuda_ver = _get_gpu_info()
    
    # PyTorch
    torch_version, torch_cuda = _get_torch_info()
    
    # Tailscale
    tailscale_info = _get_tailscale_info()
    
    # Build capabilities object
    caps = WorkerCapabilities(
        worker_id=worker_id,
        hostname=socket.gethostname(),
        ip_address=_get_ip_address(),
        cpu_count=cpu_count,
        cpu_count_logical=cpu_count_logical,
        cpu_freq_mhz=cpu_freq_mhz,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        has_gpu=has_gpu,
        gpu_count=gpu_count,
        gpus=gpus,
        total_vram_gb=total_vram,
        cuda_available=cuda_available or torch_cuda,
        tailscale=tailscale_info,
        python_version=platform.python_version(),
        os_name=platform.system(),
        os_version=platform.release(),
        torch_version=torch_version,
        torch_cuda=torch_cuda,
        sklearn_version=_get_lib_version("sklearn"),
        xgboost_version=_get_lib_version("xgboost"),
        lightgbm_version=_get_lib_version("lightgbm"),
        pycaret_version=_get_lib_version("pycaret"),
        optuna_version=_get_lib_version("optuna"),
        max_concurrency=max(1, cpu_count - 1),  # Leave 1 core for system
    )
    
    # Set tags based on capabilities
    caps.tags = caps.get_queues()
    
    # Log summary
    ts_status = "connected" if tailscale_info.connected else "not connected"
    logger.info(f"Worker {caps.worker_id}: {caps.cpu_count} CPUs, {caps.total_ram_gb}GB RAM, "
                f"{caps.gpu_count} GPUs ({caps.total_vram_gb}GB VRAM), Tailscale: {ts_status}")
    logger.info(f"Queues: {caps.tags}")
    if tailscale_info.connected:
        logger.info(f"Tailscale IP: {tailscale_info.tailscale_ip} @ {tailscale_info.tailnet}")
    
    return caps


if __name__ == "__main__":
    # Test capability probing
    caps = probe_capabilities()
    print(json.dumps(caps.to_dict(), indent=2))
