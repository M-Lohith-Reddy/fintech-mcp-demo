"""
Enhanced Training Script with Multi-Intent Testing
"""
import sys
import logging
from ml_intent_classifier import ProductionIntentClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    print("\n" + "=" * 70)
    print("FINTECH ML MODEL TRAINING - ENHANCED MULTI-INTENT")
    print("=" * 70)

    try:
        classifier = ProductionIntentClassifier()
        classifier.train()

        # ======================
        # COMPREHENSIVE TESTING
        # ======================

        print("\n" + "=" * 70)
        print("MULTI-INTENT TESTING")
        print("=" * 70)

        tests = [
            # Single intent tests
            {
                "query": "calculate gst on 10000 at 18%",
                "expected_intents": ["calculate_gst"],
                "expected_tools": 1
            },
            {
                "query": "how to register my company",
                "expected_intents": ["company_guide"],
                "expected_tools": 1
            },
            
            # Multi-intent tests
            {
                "query": "calculate gst on 5000 at 12% and show breakdown",
                "expected_intents": ["calculate_gst", "gst_breakdown"],
                "expected_tools": 2
            },
            {
                "query": "calculate gst on 10000 at 18% and also compare with 12%",
                "expected_intents": ["calculate_gst", "compare_rates"],
                "expected_tools": 2  # calculate_gst for 18%, compare with multiple rates
            },
            {
                "query": "validate gstin 29ABCDE1234F1Z5 and calculate gst on 5000 at 12%",
                "expected_intents": ["validate_gstin", "calculate_gst"],
                "expected_tools": 2
            },
            {
                "query": "show gst breakdown for 10000 at 18% along with company registration process",
                "expected_intents": ["gst_breakdown", "company_guide"],
                "expected_tools": 2
            },
            {
                "query": "calculate gst on 5000 at 5% and 12% and 18%",
                "expected_intents": ["calculate_gst"],
                "expected_tools": 3  # One for each rate
            }
        ]

        passed = 0
        failed = 0

        for idx, test in enumerate(tests, 1):
            query = test["query"]
            expected_intents = test["expected_intents"]
            expected_tools = test["expected_tools"]
            
            result = classifier.process_query(query)
            
            detected = result['intents_detected']
            tool_count = len(result['tool_calls'])
            
            # Check if all expected intents are detected
            intents_match = all(intent in detected for intent in expected_intents)
            tools_match = tool_count == expected_tools
            
            status = "✓ PASS" if (intents_match and tools_match) else "✗ FAIL"
            
            if intents_match and tools_match:
                passed += 1
            else:
                failed += 1
            
            print(f"\n[Test {idx}] {status}")
            print(f"Query: {query}")
            print(f"Expected Intents: {expected_intents}")
            print(f"Detected Intents: {detected}")
            print(f"Expected Tools: {expected_tools} | Got: {tool_count}")
            
            if not intents_match:
                missing = [i for i in expected_intents if i not in detected]
                if missing:
                    print(f"⚠ Missing intents: {missing}")
            
            if result['entities']:
                print(f"Entities: {result['entities']}")

        print("\n" + "=" * 70)
        print(f"TEST RESULTS: {passed} passed, {failed} failed out of {len(tests)}")
        print("=" * 70)

        if failed == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"⚠ {failed} tests failed. Review multi-intent detection.")

        print("\n" + "=" * 70)
        print("✓ Model saved to models/production_classifier.pkl")
        print("=" * 70)
        print("\nNext steps:")
        print("1. If tests passed: python run_client.py")
        print("2. If tests failed: Review training data and retrain")
        print("=" * 70 + "\n")

        return 0 if failed == 0 else 1

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())