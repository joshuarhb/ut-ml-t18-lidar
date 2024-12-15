import re
from collections import defaultdict

def analyze_log(filename):
    # Store latest progress for each directory
    progress = defaultdict(lambda: (0, 0))
    
    with open(filename, 'r') as f:
        for line in f:
            if 'Processing' not in line:
                continue
                
            # Extract directory name and progress
            match = re.search(r'Processing ([\d-]+_mapping_tartu): (\d+)/(\d+)', line)
            if match:
                dir_name, current, total = match.groups()
                progress[dir_name] = (int(current), int(total))
    
    # Calculate and sort by completion percentage
    results = []
    for dir_name, (current, total) in progress.items():
        percent = (current / total) * 100
        results.append((dir_name, current, total, percent))
    
    results.sort(key=lambda x: x[3])
    
    # Print results
    print("\nProcessing Progress Summary:")
    print("=" * 70)
    for dir_name, current, total, percent in results:
        print(f"{dir_name}: {current}/{total} ({percent:.1f}%)")

# Usage
analyze_log('ground_removal_53001173.err')

