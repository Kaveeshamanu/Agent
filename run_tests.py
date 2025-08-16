#!/usr/bin/env python3
"""
SteamNoodles Feedback Agent - Test Runner
Enhanced test runner with performance monitoring and detailed reporting
"""

import sys
import time
import unittest
import os
from io import StringIO
import json

def run_comprehensive_tests():
    """Run all tests with enhanced reporting and performance monitoring"""
    
    print("🍜 SteamNoodles Feedback Agent - Test Runner")
    print("=" * 60)
    
    # Setup test environment
    start_time = time.time()
    
    # Ensure test modules can be imported
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Import test module
        from test_agents import test_agents
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(sys.modules['test_agents'])
        
        # Custom test runner with enhanced output
        class VerboseTestResult(unittest.TextTestResult):
            def __init__(self, stream, descriptions, verbosity):
                super().__init__(stream, descriptions, verbosity)
                self.test_results = []
                self.performance_data = {}
            
            def startTest(self, test):
                super().startTest(test)
                self.test_start_time = time.time()
                print(f"  Running: {test._testMethodName}...", end=" ")
            
            def addSuccess(self, test):
                super().addSuccess(test)
                duration = time.time() - self.test_start_time
                self.performance_data[test._testMethodName] = duration
                print(f"✅ ({duration:.3f}s)")
                self.test_results.append({
                    'test': test._testMethodName,
                    'status': 'PASS',
                    'duration': duration
                })
            
            def addError(self, test, err):
                super().addError(test, err)
                duration = time.time() - self.test_start_time
                self.performance_data[test._testMethodName] = duration
                print(f"❌ ERROR ({duration:.3f}s)")
                self.test_results.append({
                    'test': test._testMethodName,
                    'status': 'ERROR',
                    'duration': duration,
                    'error': str(err[1])
                })
            
            def addFailure(self, test, err):
                super().addFailure(test, err)
                duration = time.time() - self.test_start_time
                self.performance_data[test._testMethodName] = duration
                print(f"❌ FAIL ({duration:.3f}s)")
                self.test_results.append({
                    'test': test._testMethodName,
                    'status': 'FAIL',
                    'duration': duration,
                    'error': str(err[1])
                })
        
        class VerboseTestRunner(unittest.TextTestRunner):
            def __init__(self):
                super().__init__(
                    stream=sys.stdout,
                    verbosity=2,
                    resultclass=VerboseTestResult
                )
        
        # Run tests
        print("\n🧪 Running Comprehensive Test Suite...")
        print("-" * 40)
        
        runner = VerboseTestRunner()
        result = runner.run(suite)
        
        # Calculate statistics
        total_time = time.time() - start_time
        total_tests = result.testsRun
        passed_tests = total_tests - len(result.failures) - len(result.errors)
        pass_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {len(result.failures)} ❌")
        print(f"Errors: {len(result.errors)} ⚠️")
        print(f"Pass Rate: {pass_rate:.1f}%")
        print(f"Total Time: {total_time:.2f} seconds")
        
        if hasattr(result, 'performance_data'):
            avg_time = sum(result.performance_data.values()) / len(result.performance_data)
            print(f"Average Test Time: {avg_time:.3f} seconds")
            slowest_test = max(result.performance_data.items(), key=lambda x: x[1])
            print(f"Slowest Test: {slowest_test[0]} ({slowest_test[1]:.3f}s)")
        
        # Show failures and errors in detail
        if result.failures:
            print("\n❌ FAILED TESTS:")
            for test, traceback in result.failures:
                print(f"  • {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\n⚠️  ERROR TESTS:")
            for test, traceback in result.errors:
                print(f"  • {test}: {str(traceback).split('\\n')[-2]}")
        
        # Performance analysis
        if hasattr(result, 'performance_data') and result.performance_data:
            print("\n⚡ PERFORMANCE ANALYSIS:")
            sorted_tests = sorted(result.performance_data.items(), key=lambda x: x[1], reverse=True)
            for test_name, duration in sorted_tests[:5]:  # Top 5 slowest
                print(f"  {test_name}: {duration:.3f}s")
        
        # Generate test report
        try:
            report = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_tests': total_tests,
                'passed': passed_tests,
                'failed': len(result.failures),
                'errors': len(result.errors),
                'pass_rate': pass_rate,
                'total_time': total_time,
                'performance_data': getattr(result, 'performance_data', {}),
                'test_results': getattr(result, 'test_results', [])
            }
            
            os.makedirs('outputs', exist_ok=True)
            with open('outputs/test_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n📄 Detailed test report saved to: outputs/test_report.json")
        except Exception as e:
            print(f"\n⚠️  Could not save test report: {e}")
        
        # Final status
        print("\n" + "=" * 60)
        if result.wasSuccessful():
            print("🎉 ALL TESTS PASSED! Ready for submission!")
        else:
            print("⚠️  Some tests failed. Please review and fix issues.")
        print("=" * 60)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Error importing test module: {e}")
        print("Make sure test_agents.py is in the current directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error running tests: {e}")
        return False

def run_quick_tests():
    """Run a subset of critical tests for quick validation"""
    print("🚀 Quick Test Mode - Running Critical Tests Only")
    print("-" * 50)
    
    try:
        from test_agents import TestFeedbackResponseAgent, TestSentimentVisualizationAgent
        
        # Create test suite with only critical tests
        suite = unittest.TestSuite()
        suite.addTest(TestFeedbackResponseAgent('test_process_feedback'))
        suite.addTest(TestFeedbackResponseAgent('test_sentiment_analysis'))
        suite.addTest(TestSentimentVisualizationAgent('test_generate_visualization'))
        suite.addTest(TestSentimentVisualizationAgent('test_date_parsing'))
        
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        if result.wasSuccessful():
            print("\n✅ Quick tests passed! Core functionality working.")
        else:
            print("\n❌ Quick tests failed. Please check core functionality.")
        
        return result.wasSuccessful()
        
    except Exception as e:
        print(f"❌ Error in quick test mode: {e}")
        return False

def main():
    """Main test runner entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SteamNoodles Feedback Agent Test Runner')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Run quick tests only (critical functionality)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    if args.quick:
        success = run_quick_tests()
    else:
        success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()