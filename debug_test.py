#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/harveybc/Documents/GitHub/predictor')

print("Starting test execution...")

try:
    from tests.unit.test_target_calculation import run_target_calculation_verification
    print("Import successful")
    
    result = run_target_calculation_verification()
    print(f"Test completed with result: {result}")
    
    # Write result to file for confirmation
    with open('/home/harveybc/Documents/GitHub/predictor/test_result.txt', 'w') as f:
        f.write(f"Target calculation test result: {result}\n")
        if result:
            f.write("SUCCESS: All target calculation tests passed!\n")
        else:
            f.write("FAILED: Some target calculation tests failed.\n")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    
    # Write error to file
    with open('/home/harveybc/Documents/GitHub/predictor/test_result.txt', 'w') as f:
        f.write(f"Test execution failed with error: {e}\n")

print("Script finished.")
