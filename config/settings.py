import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql://jarvis:strongpassword@postgres:5432/jarvisdb")
    chroma_url: str = os.getenv("CHROMA_URL", "http://chroma:8000")
    
    def validate(self) -> List[str]:
        issues = []
        if not self.postgres_url:
            issues.append("POSTGRES_URL not configured")
        if not self.chroma_url:
            issues.append("CHROMA_URL not configured")
        return issues

@dataclass
class LLMConfig:
    """Large Language Model configuration"""
    ollama_url: str = os.getenv("OLLAMA_URL", "http://192.168.0.116:11434")
    default_model: str = os.getenv("OLLAMA_MODEL", "llama3:latest")
    timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT", "30"))
    
    def validate(self) -> List[str]:
        issues = []
        if not self.ollama_url:
            issues.append("OLLAMA_URL not configured")
        return issues

@dataclass
class APIConfig:
    """External API configuration"""
    accuweather_key: str = os.getenv("ACCUWEATHER_API_KEY", "")
    mcp_proxy_url: str = os.getenv("MCP_PROXY_URL", "http://mcp-proxy:8180")
    
    def validate(self) -> List[str]:
        issues = []
        if not self.accuweather_key:
            issues.append("ACCUWEATHER_API_KEY not configured - weather features will be limited")
        return issues

@dataclass 
class MCPConfig:
    """MCP Protocol configuration - ADDED THIS SECTION"""
    proxy_url: str = os.getenv("MCP_PROXY_URL", "http://multi-mcp-proxy:9190")
    weather_server_url: str = os.getenv("MCP_WEATHER_URL", "http://weather-mcp:8182")
    echo_server_url: str = os.getenv("MCP_ECHO_URL", "http://echo-mcp:8181")
    timeout_seconds: int = int(os.getenv("MCP_TIMEOUT", "10"))
    
    def validate(self) -> List[str]:
        issues = []
        if not self.proxy_url:
            issues.append("MCP_PROXY_URL not configured")
        return issues

@dataclass
class ServiceConfig:
    """Service configuration"""
    metrics_port: int = int(os.getenv("METRICS_PORT", "8001"))
    health_port: int = int(os.getenv("HEALTH_PORT", "8005"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug_mode: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    def validate(self) -> List[str]:
        issues = []
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            issues.append(f"Invalid LOG_LEVEL: {self.log_level}")
        return issues

@dataclass
class AppSettings:
    """Main application settings"""
    # ✅ Use default_factory for mutable defaults
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)  # ✅ ADDED MCP CONFIG
    service: ServiceConfig = field(default_factory=ServiceConfig)
    
    def validate(self) -> Dict[str, Any]:
        """Validate all configuration sections"""
        all_issues = []
        all_issues.extend(self.database.validate())
        all_issues.extend(self.llm.validate())
        all_issues.extend(self.api.validate())
        all_issues.extend(self.mcp.validate())  # ✅ ADDED MCP VALIDATION
        all_issues.extend(self.service.validate())
        
        return {
            "valid": len(all_issues) == 0,
            "issues": all_issues,
            "config_summary": {
                "database_url": self.database.postgres_url,
                "ollama_url": self.llm.ollama_url,
                "mcp_proxy_url": self.mcp.proxy_url,  # ✅ ADDED MCP INFO
                "weather_api_configured": bool(self.api.accuweather_key),
                "debug_mode": self.service.debug_mode,
                "ports": {
                    "metrics": self.service.metrics_port,
                    "health": self.service.health_port
                }
            }
        }
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment information"""
        return {
            "python_path": os.getenv("PYTHONPATH", ""),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "container_mode": os.path.exists("/.dockerenv"),
            "debug_enabled": self.service.debug_mode
        }

# Global settings instance
settings = AppSettings()

# Validate settings on import with error handling
try:
    validation_result = settings.validate()
    if not validation_result["valid"]:
        print("⚠️  Configuration Issues Found:")
        for issue in validation_result["issues"]:
            print(f"   - {issue}")
        print("\n💡 Check your .env file or environment variables")
    else:
        print("✅ Configuration validated successfully")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
