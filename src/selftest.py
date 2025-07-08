import unittest
import asyncio
import time
import sys

# Import your modules (adjust these to match your container's PYTHONPATH)
from src.agents.core.main_agent import JarvisMainAgent, AutoGenLangGraphHybrid
from src.agents.specialized.weather_agent import ReliableWeatherAgent
from src.agents.specialized.routine_agent import ReliableRoutineAgent
from src.mcp.tool_discovery import YAMLToolDiscovery
from src.mcp.proxy_manager import MCPProxyManager
from src.agents.core.database import db_manager
from src.utils.config_loader import ConfigLoader
from src.agents.core.a2a_protocol import a2a_registry, A2AAgentCard, A2ASkill

def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

class TestMainAgenticWorkflow(unittest.TestCase):
    def test_agentic_chat_greeting(self):
        agent = JarvisMainAgent()
        result = run_async(agent.agentic_chat("Hello!"))
        self.assertIn("hello", result.lower())

    def test_agentic_chat_weather(self):
        agent = JarvisMainAgent()
        result = run_async(agent.agentic_chat("What's the weather in Brussels?"))
        self.assertTrue("weather" in result.lower() or "brussels" in result.lower())

    def test_agentic_chat_unknown(self):
        agent = JarvisMainAgent()
        result = run_async(agent.agentic_chat("Tell me a joke about quantum physics."))
        self.assertIsInstance(result, str)
        self.assertNotEqual(result.strip(), "")

    def test_system_status(self):
        hybrid = AutoGenLangGraphHybrid()
        status = hybrid.get_system_status()
        self.assertEqual(status.get("status"), "healthy")
        
        # FIX: Handle both dict and int for agents
        agents = status.get("agents")
        if isinstance(agents, dict):
            agent_count = agents.get("count", 0)
        else:
            agent_count = agents if isinstance(agents, int) else 0
        self.assertGreaterEqual(agent_count, 2)
        
        # FIX: Check nested workflow.initialized instead of top-level
        workflow_info = status.get("workflow", {})
        self.assertTrue(workflow_info.get("initialized", False))


class TestSpecializedAgents(unittest.TestCase):
    def test_weather_agent_demo(self):
        agent = ReliableWeatherAgent()
        agent.demo_mode = True
        result = run_async(agent.process_message("Weather in Paris", ctx=None))
        self.assertIn("paris", result.lower())
        self.assertIn("demo", result.lower())

    def test_routine_agent_basic(self):
        agent = ReliableRoutineAgent()
        # Provide a context with no sender to ensure it does not crash
        class DummyCtx:
            pass
        result = run_async(agent.process_message("Create a morning routine", ctx=DummyCtx()))
        self.assertIsInstance(result, str)

class TestDatabaseManager(unittest.TestCase):
    def test_log_event(self):
        ok = db_manager.log_event("INFO", "Self-test log event", meta={"test": True})
        self.assertTrue(ok)

    def test_register_and_heartbeat(self):
        agent_id = "selftest-agent"
        registered = db_manager.register_agent(agent_id, "test", {"cap": True}, "Selftest agent")
        self.assertTrue(registered)
        heartbeat = db_manager.update_agent_heartbeat(agent_id)
        self.assertTrue(heartbeat)

    def test_agent_context(self):
        agent_id = "selftest-agent"
        db_manager.store_agent_interaction(agent_id, "test", {"msg": "context test"})
        context = db_manager.get_agent_context(agent_id, limit=3)
        self.assertIsInstance(context, list)

    def test_active_agents(self):
        agents = db_manager.get_active_agents()
        self.assertIsInstance(agents, list)

    def test_health_check(self):
        health = db_manager.health_check()
        self.assertEqual(health.get("status"), "healthy", f"DB health error: {health.get('error')}")
        self.assertIn("database_stats", health)

class TestToolDiscoveryAndProxy(unittest.TestCase):
    def test_yaml_tool_discovery_init_and_tools(self):
        discovery = YAMLToolDiscovery("config/mcp_tools.yaml")
        ok = run_async(discovery.initialize())
        self.assertTrue(ok)
        tools = run_async(discovery.get_tools_for_agent("weather"))
        self.assertIsInstance(tools, list)
        # Use the actual tool name as defined in your YAML (likely 'get_weather')
        self.assertTrue(any(t.name == "get_weather" for t in tools))

    def test_mcp_proxy_health(self):
        proxy = MCPProxyManager("config/mcp_tools.yaml")
        health = run_async(proxy.check_proxy_health())
        self.assertIn(health.get("status"), ["healthy", "unhealthy"])

    def test_mcp_tool_call(self):
        discovery = YAMLToolDiscovery("config/mcp_tools.yaml")
        run_async(discovery.initialize())
        tools = run_async(discovery.get_tools_for_agent("weather"))
        if tools:
            tool = tools[0]
            # Try calling with dummy params; expect either a valid string or an error message
            result = run_async(tool.arun({"location": "Brussels"}))
            self.assertIsInstance(result, str)

class TestConfigLoader(unittest.TestCase):
    def test_config_loader_yaml(self):
        config = ConfigLoader.load_yaml("config/mcp_tools.yaml", use_cache=True)
        self.assertIsInstance(config, dict)

    def test_config_loader_json(self):
        config = ConfigLoader.load_json("config/mcp-servers.json", use_cache=True)
        self.assertIsInstance(config, dict)

    def test_config_validation(self):
        from config.settings import settings
        validation = settings.validate()
        self.assertTrue(validation["valid"])

class TestA2AProtocol(unittest.TestCase):
    def test_a2a_registry(self):
        # Should have at least weather and routine agents registered
        agents = a2a_registry.agents
        self.assertGreaterEqual(len(agents), 2)
        for agent_id, card in agents.items():
            card_dict = card.to_dict()
            self.assertIn("agent_id", card_dict)
            self.assertIn("skills", card_dict)

class TestEndToEndIntegration(unittest.TestCase):
    def test_end_to_end_weather_query(self):
        agent = JarvisMainAgent()
        result = run_async(agent.agentic_chat("What is the weather forecast for tomorrow in Ghent?"))
        self.assertIn("ghent", result.lower())

    def test_end_to_end_routine_creation(self):
        agent = JarvisMainAgent()
        result = run_async(agent.agentic_chat("Remind me to take medicine every day at 8am"))
        self.assertTrue(isinstance(result, str) and len(result) > 0)

    def test_error_handling(self):
        agent = JarvisMainAgent()
        # Send a deliberately malformed input
        result = run_async(agent.agentic_chat(None))
        self.assertTrue("error" in result.lower() or isinstance(result, str))

if __name__ == "__main__":
    print("Running comprehensive self-tests for Jarvis AI system...")
    start = time.time()
    result = unittest.main(verbosity=2, exit=False)
    duration = time.time() - start
    print(f"\nTotal time: {duration:.2f} seconds")
    sys.exit(0 if result.result.wasSuccessful() else 1)