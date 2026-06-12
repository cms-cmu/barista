#!/usr/bin/env python3
"""
Simple memory profiling script using memory_profiler

Usage: 
1. Make sure memory_profiler is installed: pip install memory-profiler
2. Run: python -m memory_profiler debug_processor_memory.py your_input_file.root
3. Or run: mprof run debug_processor_memory.py your_input_file.root && mprof plot
"""

import sys
import os
import psutil
import time
import gc
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

def monitor_memory_simple():
    """Simple memory monitoring function"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # MB

def main():
    if len(sys.argv) < 2:
        print("Usage: python profile_memory.py <input_file.root>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    print(f"Memory profiling processor with file: {input_file}")
    print(f"Initial memory: {monitor_memory_simple():.1f} MB")
    
    try:
        # Import and setup
        print("Importing modules...")
        start_mem = monitor_memory_simple()
        
        from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
        from python.analysis.processors.processor_HH4b import analysis
        
        after_imports = monitor_memory_simple()
        print(f"After imports: {after_imports:.1f} MB (+{after_imports-start_mem:.1f} MB)")
        
        # Create processor
        print("Creating processor...")
        processor_instance = analysis(
            fill_histograms=False,  # Disable histograms for debugging
            run_systematics=[],     # No systematics for debugging
        )
        
        after_processor = monitor_memory_simple()
        print(f"After processor creation: {after_processor:.1f} MB (+{after_processor-after_imports:.1f} MB)")
        
        # Load a small number of events
        print("Loading events...")
        events = NanoEventsFactory.from_root(
            {input_file: "Events"},
            schemaclass=NanoAODSchema,
            entry_stop=100  # Just 100 events for debugging
        ).events()
        
        after_events = monitor_memory_simple()
        print(f"After loading 100 events: {after_events:.1f} MB (+{after_events-after_processor:.1f} MB)")
        
        # Process events
        print("Processing events...")
        process_start = monitor_memory_simple()
        
        result = processor_instance.process(events)
        
        after_processing = monitor_memory_simple()
        print(f"After processing: {after_processing:.1f} MB (+{after_processing-process_start:.1f} MB)")
        
        # Force garbage collection
        gc.collect()
        after_gc = monitor_memory_simple()
        print(f"After garbage collection: {after_gc:.1f} MB (freed {after_processing-after_gc:.1f} MB)")
        
        print("\nMemory profiling completed successfully!")
        print("\nFor time-series memory profiling:")
        print("mprof run your_script.py && mprof plot")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()
        
        final_mem = monitor_memory_simple()
        print(f"Memory at error: {final_mem:.1f} MB")

if __name__ == "__main__":
    main()
