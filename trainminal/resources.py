"""Resource monitoring for GPU, CPU, and RAM."""

import time
from typing import Dict, List, Optional, Tuple
import psutil

try:
    # Try nvidia-ml-py first (newer package)
    try:
        import pynvml
        NVML_AVAILABLE = True
    except ImportError:
        # Fallback to nvidia-ml-py
        import pynvml
        NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class ResourceMonitor:
    """Monitor system resources including GPU, CPU, and RAM."""
    
    def __init__(self):
        self.nvml_initialized = False
        self.gpu_count = 0
        self._init_nvml()
    
    def _init_nvml(self):
        """Initialize NVIDIA Management Library."""
        if not NVML_AVAILABLE:
            return
        
        try:
            pynvml.nvmlInit()
            self.nvml_initialized = True
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        except Exception:
            self.nvml_initialized = False
            self.gpu_count = 0
    
    def get_gpu_info(self) -> List[Dict[str, any]]:
        """Get information about all available GPUs."""
        if not self.nvml_initialized:
            return []
        
        gpu_info = []
        try:
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU name
                name_bytes = pynvml.nvmlDeviceGetName(handle)
                # Handle both bytes and string returns (different pynvml versions)
                name = name_bytes.decode('utf-8') if isinstance(name_bytes, bytes) else name_bytes
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory = mem_info.total
                used_memory = mem_info.used
                free_memory = mem_info.free
                
                # Get utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                
                # Get temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except:
                    temp = None
                
                # Get power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                except:
                    power = None
                
                gpu_info.append({
                    'index': i,
                    'name': name,
                    'utilization': gpu_util,
                    'memory_total': total_memory,
                    'memory_used': used_memory,
                    'memory_free': free_memory,
                    'memory_used_percent': (used_memory / total_memory) * 100 if total_memory > 0 else 0,
                    'temperature': temp,
                    'power': power,
                })
        except Exception as e:
            # If there's an error, return empty list
            return []
        
        return gpu_info
    
    def get_cpu_info(self) -> Dict[str, any]:
        """Get CPU utilization information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            return {
                'percent': cpu_percent,
                'count': cpu_count,
                'per_core': cpu_per_core,
            }
        except Exception:
            return {
                'percent': 0.0,
                'count': 0,
                'per_core': [],
            }
    
    def get_ram_info(self) -> Dict[str, any]:
        """Get RAM usage information."""
        try:
            mem = psutil.virtual_memory()
            return {
                'total': mem.total,
                'used': mem.used,
                'available': mem.available,
                'percent': mem.percent,
                'free': mem.free,
            }
        except Exception:
            return {
                'total': 0,
                'used': 0,
                'available': 0,
                'percent': 0.0,
                'free': 0,
            }
    
    def get_all_resources(self) -> Dict[str, any]:
        """Get all resource information."""
        return {
            'gpus': self.get_gpu_info(),
            'cpu': self.get_cpu_info(),
            'ram': self.get_ram_info(),
            'timestamp': time.time(),
        }
    
    def has_gpu(self) -> bool:
        """Check if GPU monitoring is available."""
        return self.nvml_initialized and self.gpu_count > 0
    
    def cleanup(self):
        """Clean up resources."""
        if self.nvml_initialized:
            try:
                # pynvml doesn't have a cleanup function, but we can mark as not initialized
                self.nvml_initialized = False
            except Exception:
                pass

