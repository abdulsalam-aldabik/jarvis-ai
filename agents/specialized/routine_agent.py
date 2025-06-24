from autogen_core import MessageContext
from agents.core.base_agent import AutoGenBaseAgent

class ReliableRoutineAgent(AutoGenBaseAgent):
    """AutoGen-native routine agent"""
    
    def __init__(self):
        super().__init__("routine", "Routine planning using AutoGen")
    
    async def process_message(self, message: str, ctx: MessageContext) -> str:
        """Process routine requests"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["create", "make", "new"]):
            return self._suggest_routine_creation(message)
        elif any(word in message_lower for word in ["improve", "optimize", "better"]):
            return self._suggest_optimization()
        else:
            return self._general_routine_help()
    
    def _suggest_routine_creation(self, message: str) -> str:
        """Suggest routine creation"""
        if "morning" in message.lower():
            return ("Here's a simple morning routine:\n"
                   "• 7:00 AM - Wake up and stretch\n"
                   "• 7:15 AM - Exercise or walk\n" 
                   "• 7:45 AM - Breakfast and planning\n"
                   "• 8:30 AM - Start work/day activities")
        elif "evening" in message.lower():
            return ("Here's an evening routine:\n"
                   "• 9:00 PM - Review the day\n"
                   "• 9:30 PM - Prepare for tomorrow\n"
                   "• 10:00 PM - Relaxation time\n"
                   "• 10:30 PM - Wind down for sleep")
        else:
            return ("I can help you create routines! Try asking about:\n"
                   "• Morning routines\n"
                   "• Evening routines\n" 
                   "• Work routines\n"
                   "What type interests you?")
    
    def _suggest_optimization(self) -> str:
        """Suggest routine optimization"""
        return ("To optimize your routine:\n"
               "• Track what works best for you\n"
               "• Adjust timing based on your energy\n"
               "• Build in flexibility for unexpected events\n"
               "• Review and adjust weekly")
    
    def _general_routine_help(self) -> str:
        """General routine guidance"""
        return ("I can help you with routine planning!\n"
               "• Creating new routines\n"
               "• Optimizing existing ones\n"
               "• Time management tips\n"
               "What would you like to work on?")
