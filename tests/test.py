# tests/test_intent_classifier.py
import pytest
import asyncio
from src.intent_classifier import IntentClassifier

@pytest.fixture
async def classifier():
    classifier = IntentClassifier()
    await classifier.initialize()
    return classifier

@pytest.mark.asyncio
async def test_weather_intent_classification(classifier):
    """Test weather intent classification accuracy"""
    test_cases = [
        ("What's the weather like?", "weather_query"),
        ("Is it going to rain today?", "weather_query"),
        ("Current temperature", "weather_query"),
    ]
    
    for message, expected_intent in test_cases:
        intent, confidence, agent = await classifier.classify_intent(message)
        assert intent == expected_intent
        assert confidence > 0.6
        assert agent == "weather"

@pytest.mark.asyncio
async def test_routine_intent_classification(classifier):
    """Test routine intent classification accuracy"""
    test_cases = [
        ("Create a morning routine", "routine_create"),
        ("Help me build a workout schedule", "routine_create"),
        ("Show my routines", "routine_manage"),
    ]
    
    for message, expected_intent in test_cases:
        intent, confidence, agent = await classifier.classify_intent(message)
        assert intent == expected_intent
        assert confidence > 0.6
        assert agent == "routine"

@pytest.mark.asyncio
async def test_intent_confidence_thresholds(classifier):
    """Test that low confidence messages default to general conversation"""
    unclear_message = "xyz abc random"
    intent, confidence, agent = await classifier.classify_intent(unclear_message)
    assert intent == "general_conversation"
    assert agent == "orchestrator"
