# Memory Testing and Monitoring Tools

This directory contains utilities for monitoring and limiting memory usage during analysis workflows to prevent system hangs and crashes.

## Quick Start

**Prevent system hangs by wrapping any command:**
```bash
# From project root directory
src/scripts/memory/memory_limited_run.sh --max-memory 4000 -- YOUR_COMMAND_HERE
```

## Tools Overview

### 🛡️ **memory_limited_run.sh** - Universal Memory Guard
**Primary tool for preventing system hangs**

Monitors any command and kills it if memory usage exceeds the threshold.

```bash
# Basic usage
src/scripts/memory/memory_limited_run.sh --max-memory 4000 -- your_command

# With custom settings
src/scripts/memory/memory_limited_run.sh \
  --max-memory 3000 \
  --check-interval 5 \
  --kill-threshold 0.8 \
  --log-file my_analysis.log \
  -- ./run_container python your_script.py --your-args
```

**Options:**

- `--max-memory MB` - Maximum memory in MB (default: 2000)
- `--check-interval SEC` - Check frequency in seconds (default: 2)  
- `--kill-threshold RATIO` - Kill at ratio of max memory (default: 0.8)
- `--log-file FILE` - Log file name (default: memory_monitor.log)

### 🐍 **memory_monitor.py** - Python-Specific Monitor
**For Python scripts with detailed monitoring**

```bash
# Monitor Python scripts
src/scripts/memory/memory_monitor.py --max-memory 4000 python your_analysis.py

# With custom settings
src/scripts/memory/memory_monitor.py \
  --max-memory 3000 \
  --check-interval 10 \
  --log-file python_memory.log \
  python your_script.py --your-args
```

### 📊 **profile_memory_simple.py** - Memory Profiling
**For understanding memory usage patterns**

```bash
# Profile a specific ROOT file processing
src/scripts/memory/profile_memory_simple.py /path/to/your/input.root

# Shows memory usage at different stages:
# - Imports
# - Processor creation  
# - Event loading
# - Processing
# - Garbage collection
```

### 🔍 **debug_memory.py** - Advanced Memory Analysis
**For deep debugging of memory issues**

Contains advanced tools for:
- Object-level memory analysis
- Memory leak detection
- Awkward array size inspection
- Memory hotspot identification

```python
# Use in your Python code
from src.scripts.memory.debug_memory import MemoryMonitor

monitor = MemoryMonitor()
monitor.checkpoint("before_analysis")
# ... your analysis code ...
monitor.checkpoint("after_analysis")
monitor.summary()
```

### ✅ **memory_test.py** - Automated Memory Testing & Validation
**For automated testing and CI/CD memory validation**

Runs scripts with memory profiling and validates they stay within memory thresholds.

```bash
# Test that a script stays within memory limits
src/scripts/memory/memory_test.py \
  --threshold 3000 \
  --tolerance 10 \
  --output test_results \
  --script your_analysis.py --your-args

# Returns exit code 0 if within threshold, 1 if exceeded
# Automatically generates memory plots and extracts peak usage
```

**Key Features:**
- Automated `mprof` execution with plot generation
- Peak memory extraction and threshold validation
- Tolerance checking (threshold ± percentage)
- Proper exit codes for CI/CD integration
- Output files: `test_results.dat`, `test_results.png`

## Common Use Cases

### 🚀 **Testing Container Workflows**
```bash
src/scripts/memory/memory_limited_run.sh --max-memory 6000 -- \
  ./run_container source coffea4bees/scripts/memory_test.sh
```

### 🐍 **Testing Snakemake Workflows**
```bash
# Test full workflow
src/scripts/memory/memory_limited_run.sh --max-memory 8000 -- \
  snakemake -s coffea4bees/workflows/Snakefile_mixdata_closure

# Test specific rule
src/scripts/memory/memory_limited_run.sh --max-memory 4000 -- \
  snakemake -s coffea4bees/workflows/Snakefile_mixdata_closure mixed_bkg_tt
```

### 🔬 **Debugging Memory Issues**
```bash
# 1. First, profile to understand usage patterns
src/scripts/memory/profile_memory_simple.py your_input.root

# 2. Then run with conservative limits
src/scripts/memory/memory_limited_run.sh --max-memory 2000 --check-interval 5 -- \
  your_problematic_command

# 3. Check the logs
tail -f memory_monitor.log
```

### ✅ **Automated Memory Testing**
```bash
# Test that analysis stays within memory limits (for CI/CD)
src/scripts/memory/memory_test.py \
  --threshold 4000 \
  --tolerance 10 \
  --output analysis_test \
  --script coffea4bees/analysis/processors/processor_HH4b.py your_input.root

# Check if test passed (exit code 0 = success, 1 = failure)
echo "Test result: $?"

# View generated memory plot
ls analysis_test.png
```

## Memory Sizing Guidelines

**For systems with 11GB RAM (like cmslpc):**

| Workers | Memory per Worker | Total Memory | Safety Level |
|---------|------------------|--------------|--------------|
| 1       | 8GB             | 8GB          | ✅ **Safe** |
| 2       | 4GB             | 8GB          | ✅ **Safe** |
| 3       | 3GB             | 9GB          | ⚠️ **Risky** |
| 4+      | Any             | 10GB+        | ❌ **Unsafe** |

**Recommended starting points:**
- **Development/Testing**: `--max-memory 2000` (2GB)
- **Small analysis**: `--max-memory 4000` (4GB)  
- **Full analysis**: `--max-memory 6000` (6GB)
- **Maximum safe**: `--max-memory 8000` (8GB)

## Monitoring and Logs

### Real-time Monitoring
```bash
# Watch memory usage live
tail -f memory_monitor.log

# Filter for just memory usage
grep "Memory usage" memory_monitor.log
```

### Log Analysis
```bash
# Find memory spikes
grep "MEMORY THRESHOLD EXCEEDED" memory_monitor.log

# See memory progression
grep "Memory usage:" memory_monitor.log | tail -20
```

## Integration with Analysis Workflows

### In Configuration Files
Update your analysis configs to use fewer workers:

```yaml
# coffea4bees/analysis/metadata/your_config.yml
runner:
  workers: 1              # Start with 1 for debugging
  condor_memory: 8GB      # Allow more memory per worker
  maxchunks: 1           # Limit chunks for testing
```

### In Processor Development
Add memory monitoring to your processor:

```python
# In your processor
from src.scripts.memory.debug_memory import MemoryMonitor

class YourProcessor:
    def __init__(self):
        self.memory_monitor = MemoryMonitor()
    
    def process(self, events):
        self.memory_monitor.checkpoint("start_processing")
        # ... your processing ...
        self.memory_monitor.checkpoint("end_processing")
```

## Troubleshooting

### "Command not found"
Make sure you're running from the project root directory:
```bash
# Run from here
/path/to//$ src/scripts/memory/memory_limited_run.sh ...

# NOT from here  
/path/to//src/scripts/memory/$ ./memory_limited_run.sh ...
```

### "Permission denied"
Make scripts executable:
```bash
chmod +x src/scripts/memory/*.sh src/scripts/memory/*.py
```

### Still Getting System Hangs
- Reduce `--max-memory` further
- Decrease `--check-interval` to 2-5 seconds
- Check that you're not running multiple memory-intensive processes
- Verify your system has enough swap space

## Related Documentation

- **MEMORY_TESTING_GUIDE.md** - Quick reference commands
- **MEMORY_DEBUG_GUIDE.md** - Comprehensive debugging strategies
- **src/README.md** - Overall src/ directory structure

## Contributing

When adding new memory tools:
1. Follow the existing naming convention
2. Include `--help` documentation
3. Add examples to this README
4. Test with both small and large workloads
