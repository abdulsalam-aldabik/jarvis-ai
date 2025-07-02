"""
Enhanced configuration utilities for YAML-driven MCP integration with Jarvis infrastructure
Integrates with structured logging and sophisticated error handling
"""
import yaml
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass

from src.agents.core.logging_config import log_structured


@dataclass
class ConfigMetadata:
    """Configuration metadata for caching and validation"""
    path: str
    loaded_at: float
    file_size: int
    modified_time: float
    config_type: str
    is_valid: bool = True


class ConfigLoader:
    """
    Enhanced YAML/JSON configuration loader with sophisticated caching and error handling
    Integrated with Jarvis structured logging and MCP infrastructure
    """
    
    _cache: Dict[str, Dict[str, Any]] = {}
    _metadata: Dict[str, ConfigMetadata] = {}
    _cache_ttl: int = 300  # 5 minutes cache TTL
    
    @classmethod
    def load_yaml(cls, config_path: str, use_cache: bool = True, validate_schema: bool = True) -> Dict[str, Any]:
        """
        Load YAML configuration with enhanced caching and validation
        
        Args:
            config_path: Path to YAML configuration file
            use_cache: Whether to use cached configuration
            validate_schema: Whether to validate configuration schema
            
        Returns:
            Dictionary containing configuration data
        """
        try:
            config_file = Path(config_path)
            cache_key = str(config_file.resolve())
            
            # Check cache first if enabled
            if use_cache and cls._is_cache_valid(cache_key):
                log_structured("config_cache_hit", config_path=config_path)
                return cls._cache[cache_key]
            
            if not config_file.exists():
                log_structured("config_file_not_found", config_path=config_path)
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load and parse YAML
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Validate configuration if requested
            if validate_schema:
                cls._validate_mcp_config(config, config_path)
            
            # Cache the configuration with metadata
            file_stat = config_file.stat()
            cls._cache[cache_key] = config
            cls._metadata[cache_key] = ConfigMetadata(
                path=config_path,
                loaded_at=time.time(),
                file_size=file_stat.st_size,
                modified_time=file_stat.st_mtime,
                config_type="yaml",
                is_valid=True
            )
            
            log_structured("config_loaded_successfully", 
                          config_path=config_path, 
                          config_type="yaml",
                          file_size=file_stat.st_size,
                          cache_enabled=use_cache)
            
            return config
            
        except yaml.YAMLError as ye:
            log_structured("config_yaml_parse_error", 
                          config_path=config_path, 
                          error=str(ye),
                          line=getattr(ye, 'problem_mark', {}).get('line', 'unknown'))
            raise ValueError(f"YAML parsing error in {config_path}: {str(ye)}")
            
        except Exception as e:
            log_structured("config_load_failed", 
                          config_path=config_path, 
                          error=str(e),
                          error_type=type(e).__name__)
            raise Exception(f"Failed to load configuration from {config_path}: {str(e)}")
    
    @classmethod
    def load_json(cls, config_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load JSON configuration with caching (for MCP proxy config)
        
        Args:
            config_path: Path to JSON configuration file
            use_cache: Whether to use cached configuration
            
        Returns:
            Dictionary containing configuration data
        """
        try:
            config_file = Path(config_path)
            cache_key = str(config_file.resolve())
            
            # Check cache first if enabled
            if use_cache and cls._is_cache_valid(cache_key):
                log_structured("config_cache_hit", config_path=config_path)
                return cls._cache[cache_key]
            
            if not config_file.exists():
                log_structured("config_file_not_found", config_path=config_path)
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load and parse JSON
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Cache the configuration with metadata
            file_stat = config_file.stat()
            cls._cache[cache_key] = config
            cls._metadata[cache_key] = ConfigMetadata(
                path=config_path,
                loaded_at=time.time(),
                file_size=file_stat.st_size,
                modified_time=file_stat.st_mtime,
                config_type="json",
                is_valid=True
            )
            
            log_structured("config_loaded_successfully", 
                          config_path=config_path, 
                          config_type="json",
                          file_size=file_stat.st_size)
            
            return config
            
        except json.JSONDecodeError as je:
            log_structured("config_json_parse_error", 
                          config_path=config_path, 
                          error=str(je),
                          line=getattr(je, 'lineno', 'unknown'))
            raise ValueError(f"JSON parsing error in {config_path}: {str(je)}")
            
        except Exception as e:
            log_structured("config_load_failed", 
                          config_path=config_path, 
                          error=str(e),
                          error_type=type(e).__name__)
            raise Exception(f"Failed to load configuration from {config_path}: {str(e)}")
    
    @classmethod
    def load_mcp_config(cls, config_path: str = "config/mcp_tools.yaml") -> Dict[str, Any]:
        """
        Load MCP tools configuration with environment variable substitution
        
        Args:
            config_path: Path to MCP tools YAML configuration
            
        Returns:
            MCP configuration with environment variables resolved
        """
        try:
            config = cls.load_yaml(config_path, validate_schema=True)
            
            # Substitute environment variables
            config = cls._substitute_env_vars(config)
            
            log_structured("mcp_config_loaded", 
                          config_path=config_path,
                          proxy_host=config.get("mcp_proxy", {}).get("server", {}).get("host", "unknown"),
                          agent_count=len(config.get("agent_tools", {})),
                          client_count=len(config.get("mcp_proxy", {}).get("clients", {})))
            
            return config
            
        except Exception as e:
            log_structured("mcp_config_load_failed", 
                          config_path=config_path, error=str(e))
            # Return minimal default configuration
            return cls._get_default_mcp_config()
    
    @classmethod
    def reload_yaml(cls, config_path: str) -> Dict[str, Any]:
        """Reload YAML configuration (bypass cache)"""
        config_file = Path(config_path)
        cache_key = str(config_file.resolve())
        
        if cache_key in cls._cache:
            del cls._cache[cache_key]
            log_structured("config_cache_cleared", config_path=config_path)
        
        if cache_key in cls._metadata:
            del cls._metadata[cache_key]
        
        return cls.load_yaml(config_path)
    
    @classmethod
    def reload_json(cls, config_path: str) -> Dict[str, Any]:
        """Reload JSON configuration (bypass cache)"""
        config_file = Path(config_path)
        cache_key = str(config_file.resolve())
        
        if cache_key in cls._cache:
            del cls._cache[cache_key]
            log_structured("config_cache_cleared", config_path=config_path)
        
        if cache_key in cls._metadata:
            del cls._metadata[cache_key]
        
        return cls.load_json(config_path)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all configuration cache"""
        cache_count = len(cls._cache)
        cls._cache.clear()
        cls._metadata.clear()
        log_structured("config_cache_cleared_all", cached_configs=cache_count)
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        current_time = time.time()
        valid_entries = sum(1 for key in cls._cache.keys() if cls._is_cache_valid(key))
        
        stats = {
            "total_cached_configs": len(cls._cache),
            "valid_cached_configs": valid_entries,
            "expired_cached_configs": len(cls._cache) - valid_entries,
            "cache_ttl_seconds": cls._cache_ttl,
            "memory_usage_estimate": sum(len(str(config)) for config in cls._cache.values())
        }
        
        log_structured("config_cache_stats", **stats)
        return stats
    
    @classmethod
    def _is_cache_valid(cls, cache_key: str) -> bool:
        """Check if cached configuration is still valid"""
        if cache_key not in cls._cache or cache_key not in cls._metadata:
            return False
        
        metadata = cls._metadata[cache_key]
        current_time = time.time()
        
        # Check TTL
        if current_time - metadata.loaded_at > cls._cache_ttl:
            return False
        
        # Check if file was modified
        try:
            config_file = Path(metadata.path)
            if config_file.exists():
                current_mtime = config_file.stat().st_mtime
                if current_mtime > metadata.modified_time:
                    return False
        except Exception:
            return False
        
        return True
    
    @classmethod
    def _validate_mcp_config(cls, config: Dict[str, Any], config_path: str) -> None:
        """Validate MCP configuration structure"""
        try:
            # Check required top-level sections
            required_sections = ["mcp_proxy", "agent_tools"]
            for section in required_sections:
                if section not in config:
                    log_structured("config_validation_warning", 
                                  config_path=config_path,
                                  missing_section=section)
            
            # Validate proxy configuration
            if "mcp_proxy" in config :
                proxy_config = config["mcp_proxy"]
                if "server" in proxy_config:
                    server_config = proxy_config["server"]
                    required_server_fields = ["name", "host", "port"]
                    for field in required_server_fields:
                        if field not in server_config:
                            log_structured("config_validation_warning",
                                          config_path=config_path,
                                          missing_proxy_field=field)
            
            # Validate agent tools configuration
            if "agent_tools" in config:
                agent_tools = config["agent_tools"]
                for agent_id, agent_config in agent_tools.items():
                    if "class_name" not in agent_config:
                        log_structured("config_validation_warning",
                                      config_path=config_path,
                                      agent_missing_class=agent_id)
            
            log_structured("config_validation_completed", 
                          config_path=config_path, status="valid")
            
        except Exception as e:
            log_structured("config_validation_failed", 
                          config_path=config_path, error=str(e))
    
    @classmethod
    def _substitute_env_vars(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration"""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {key: substitute_recursive(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                default_value = None
                
                # Support default values: ${VAR:default}
                if ":" in env_var:
                    env_var, default_value = env_var.split(":", 1)
                
                return os.getenv(env_var, default_value or obj)
            else:
                return obj
        
        return substitute_recursive(config)
    
    @classmethod
    def _get_default_mcp_config(cls) -> Dict[str, Any]:
        """Get minimal default MCP configuration for fallback"""
        return {
            "mcp_proxy": {
                "server": {
                    "name": "Jarvis MCP Proxy",
                    "host": "localhost",
                    "port": 9190,
                    "base_url": "http://localhost:9190"
                },
                "clients": {}
            },
            "agent_tools": {},
            "behavior_integration": {
                "enabled": False
            },
            "proxy_config": {
                "log_level": "info",
                "http_enabled": True
            }
        }
    
    @classmethod
    def get_config_path(cls, config_name: str, env: Optional[str] = None) -> str:
        """
        Get configuration path with environment-specific overrides
        
        Args:
            config_name: Base configuration name (e.g., "mcp_tools")
            env: Environment name (e.g., "dev", "prod")
            
        Returns:
            Path to configuration file
        """
        base_path = Path("config")
        
        if env:
            env_specific_path = base_path / f"{config_name}.{env}.yaml"
            if env_specific_path.exists():
                return str(env_specific_path)
        
        default_path = base_path / f"{config_name}.yaml"
        return str(default_path)
    
    @classmethod
    def load_environment_config(cls, config_name: str, env: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration with environment-specific overrides
        
        Args:
            config_name: Configuration name
            env: Environment (defaults to ENVIRONMENT env var)
            
        Returns:
            Configuration with environment overrides applied
        """
        if env is None:
            env = os.getenv("ENVIRONMENT", "dev")
        
        config_path = cls.get_config_path(config_name, env)
        
        log_structured("config_loading_environment", 
                      config_name=config_name, 
                      environment=env,
                      config_path=config_path)
        
        return cls.load_yaml(config_path)
