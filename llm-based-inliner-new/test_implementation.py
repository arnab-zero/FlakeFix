#!/usr/bin/env python3
"""
Quick test script to verify the enhanced implementation
"""

import json
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from method_inliner import MethodInliner, MethodInfo, InlineResult
        from prompts import (
            get_inline_prompt,
            get_consistency_check_prompt,
            get_rewrite_prompt,
            get_comment_removal_prompt
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_prompt_generation():
    """Test that prompts can be generated"""
    print("\nTesting prompt generation...")
    try:
        from prompts import (
            get_inline_prompt,
            get_consistency_check_prompt,
            get_rewrite_prompt
        )
        
        # Test inline prompt
        inline_prompt = get_inline_prompt(
            "public void test() { foo(); }",
            "com.example.Foo.foo",
            "public void foo() { System.out.println(\"test\"); }"
        )
        assert len(inline_prompt) > 0
        assert "com.example.Foo.foo" in inline_prompt
        print("  ✓ Inline prompt generated")
        
        # Test consistency check prompt
        consistency_prompt = get_consistency_check_prompt(
            "public void test() { foo(); }",
            [{"fqn": "com.example.Foo.foo", "body": "public void foo() {}"}],
            "public void test() { System.out.println(\"test\"); }"
        )
        assert len(consistency_prompt) > 0
        assert "is_consistent" in consistency_prompt
        print("  ✓ Consistency check prompt generated")
        
        # Test rewrite prompt
        rewrite_prompt = get_rewrite_prompt(
            "public void test() { foo(); }",
            [{"fqn": "com.example.Foo.foo", "body": "public void foo() {}"}],
            "public void test() { bar(); }",
            ["Method signature changed", "Wrong method called"]
        )
        assert len(rewrite_prompt) > 0
        assert "PROBLEMS FOUND" in rewrite_prompt
        print("  ✓ Rewrite prompt generated")
        
        print("✓ All prompts generated successfully")
        return True
    except Exception as e:
        print(f"✗ Prompt generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_method_info():
    """Test MethodInfo dataclass"""
    print("\nTesting MethodInfo dataclass...")
    try:
        from method_inliner import MethodInfo
        
        method = MethodInfo(
            fqn="com.example.Test.testMethod",
            name="testMethod",
            class_name="Test",
            package_name="com.example",
            body_lines=["public void testMethod() {", "    // test", "}"],
            body_text="public void testMethod() {\n    // test\n}",
            calls=["com.example.Foo.foo"]
        )
        
        assert method.fqn == "com.example.Test.testMethod"
        assert method.name == "testMethod"
        assert len(method.calls) == 1
        print("✓ MethodInfo works correctly")
        return True
    except Exception as e:
        print(f"✗ MethodInfo test failed: {e}")
        return False


def test_inliner_initialization():
    """Test that MethodInliner can be initialized"""
    print("\nTesting MethodInliner initialization...")
    try:
        from method_inliner import MethodInliner
        
        # Create with custom log file
        log_file = "./test_log.txt"
        inliner = MethodInliner(
            model_name="qwen3-coder:480b-cloud",
            log_file=log_file,
            max_retries=2
        )
        
        assert inliner.model_name == "qwen3-coder:480b-cloud"
        assert inliner.max_retries == 2
        assert Path(log_file).exists()
        
        # Clean up
        Path(log_file).unlink()
        
        print("✓ MethodInliner initialized successfully")
        return True
    except Exception as e:
        print(f"✗ MethodInliner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_file_exists():
    """Test that input file exists"""
    print("\nTesting input file...")
    
    candidates = [
        "../routingTest_call_analysis.json",
        "./routingTest_call_analysis.json"
    ]
    
    for candidate in candidates:
        if Path(candidate).exists():
            print(f"✓ Found input file: {candidate}")
            try:
                with open(candidate, 'r') as f:
                    data = json.load(f)
                assert "test_method" in data
                print(f"  ✓ Valid JSON structure")
                return True
            except Exception as e:
                print(f"  ✗ Invalid JSON: {e}")
                return False
    
    print("⚠ Input file not found (this is OK if not testing full run)")
    return True


def main():
    """Run all tests"""
    print("=" * 80)
    print("Enhanced Method Inliner - Implementation Tests")
    print("=" * 80)
    
    tests = [
        ("Imports", test_imports),
        ("Prompt Generation", test_prompt_generation),
        ("MethodInfo", test_method_info),
        ("Inliner Initialization", test_inliner_initialization),
        ("Input File", test_input_file_exists)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 80)
    print("Test Results")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Implementation is ready to use.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix before using.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
