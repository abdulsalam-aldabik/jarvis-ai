"""
Configuration utilities for YAML-driven MCP integration.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """YAML configuration loader with caching."""
    
    _cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def load_yaml(cls, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration with caching."""
        try:
            config_file = Path(config_path)
            
            # Check cache first
            if str(config_file) in cls._cache:
                return cls._cache[str(config_file)]
            
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Cache the configuration
            cls._cache[str(config_file)] = config
            return config
            
        except Exception as e:
            raise Exception(f"Failed to load configuration from {config_path}: {str(e)}")
    
    @classmethod
    def reload_yaml(cls, config_path: str) -> Dict[str, Any]:
        """Reload YAML configuration (bypass cache)."""
        config_file = Path(config_path)
        if str(config_file) in cls._cache:
            del cls._cache[str(config_file)]
        return cls.load_yaml(config_path)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear configuration cache."""
        cls._cache.clear()
