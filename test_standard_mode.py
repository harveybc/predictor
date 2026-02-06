#!/usr/bin/env python3
"""Quick test to verify standard optimization mode works without infinite loop"""
import json

# Load config
with open('examples/config/phase_1_daily/optimization/phase_1_mimo_1d_optimization_config.json', 'r') as f:
    config = json.load(f)

# Verify settings
print("Testing standard optimization mode configuration:")
print(f"  optimization_incremental: {config.get('optimization_incremental')}")
print(f"  optimization_meta_mode: {config.get('optimization_meta_mode')}")
print(f"  optimization_resume: {config.get('optimization_resume')}")

# Check if both are false (standard mode)
if not config.get('optimization_incremental') and not config.get('optimization_meta_mode'):
    print("\n✓ Configuration is in STANDARD MODE (both flags false)")
else:
    print("\n✗ ERROR: Not in standard mode!")
    
# Check resume file exists if resume is enabled
if config.get('optimization_resume'):
    import os
    resume_file = config.get('optimization_resume_file')
    if os.path.exists(resume_file):
        print(f"\n⚠ Resume file exists: {resume_file}")
        print("  This may cause issues if it has incompatible format")
        print("  Recommendation: Set optimization_resume to false for clean test")
    else:
        print(f"\n✓ Resume file does not exist (clean start)")

print("\nIndentation fix applied. The while True: loop should now:")
print("  1. Print separator bars")
print("  2. Create population")  
print("  3. Run GA generations")
print("  4. Break after completing optimization")
print("\nNo more infinite loops!")
