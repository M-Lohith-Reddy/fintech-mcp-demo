"""
Test Script for MCP Demo
Tests single and multi-intent functionality
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client.llm_service import claude_service


async def test_single_intent():
    """Test single intent detection"""
    print("\n" + "="*60)
    print("TEST 1: Single Intent - Calculate GST")
    print("="*60)
    
    query = "Calculate GST on 10000 rupees at 18%"
    print(f"Query: {query}\n")
    
    result = await claude_service.process_query(query)
    
    print(f"Intents Detected: {result['intents_detected']}")
    print(f"Multi-Intent: {result['is_multi_intent']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nTool Calls:")
    for tool_call in result['tool_calls']:
        print(f"  - {tool_call['tool']}: {tool_call.get('result', tool_call.get('error'))}")


async def test_multi_intent():
    """Test multi-intent detection"""
    print("\n" + "="*60)
    print("TEST 2: Multi-Intent - Calculate + Breakdown")
    print("="*60)
    
    query = "Calculate GST on 5000 at 12% and also show me the CGST and SGST breakdown"
    print(f"Query: {query}\n")
    
    result = await claude_service.process_query(query)
    
    print(f"Intents Detected: {result['intents_detected']}")
    print(f"Multi-Intent: {result['is_multi_intent']}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nTool Calls:")
    for tool_call in result['tool_calls']:
        print(f"  - {tool_call['tool']}")


async def test_natural_language():
    """Test natural language understanding"""
    print("\n" + "="*60)
    print("TEST 3: Natural Language")
    print("="*60)
    
    query = "Hey, I need to know the tax on twenty-five thousand at eighteen percent"
    print(f"Query: {query}\n")
    
    result = await claude_service.process_query(query)
    
    print(f"Intents Detected: {result['intents_detected']}")
    print(f"\nResponse:\n{result['response']}")


async def main():
    """Run all tests"""
    print("\n" + "#"*60)
    print("# MCP Demo - Test Suite")
    print("#"*60)
    
    try:
        await test_single_intent()
        await test_multi_intent()
        await test_natural_language()
        
        print("\n" + "#"*60)
        print("# All Tests Completed Successfully!")
        print("#"*60 + "\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())