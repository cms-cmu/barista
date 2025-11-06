# Base Class Library Architecture

This library provides the foundational components for physics data analysis, organized into logical domains for easy discovery and maintainability.

## Directory Structure

### 📊 Data Formats (`data_formats/`)
Tools for converting and manipulating different data formats commonly used in physics analysis.

- **`awkward/`** - Awkward array operations, conversions, padding, and structure inspection
- **`numpy/`** - NumPy array conversions and utilities  
- **`root/`** - Comprehensive ROOT file I/O, tree manipulation, friend trees, and data chaining

**Use when:** You need to read ROOT files, convert between array formats, or manipulate data structures.

### ⚙️ Configuration (`config/`)
Configuration management system with parsing, validation, and patching capabilities.

- Configuration loading from files
- Dynamic configuration patching
- Type-safe configuration protocols
- Extensible parser system

**Use when:** You need to manage analysis configuration, parse YAML/JSON config files, or implement configurable components.

### 🖥️ Distributed Computing (`dask/`)
Utilities for distributed and parallel computing using Dask.

- Delayed computation wrappers
- Distributed histogram operations
- Awkward array integration with Dask

**Use when:** You need to scale computations across multiple cores or machines.

### 💾 Storage (`storage/`)
Abstractions for working with different storage systems.

- **`eos.py`** - EOS (CERN file system) and XRootD integration
- Local and remote file system abstractions
- Path utilities and file operations

**Use when:** You need to access files on EOS, handle remote storage, or work with distributed file systems.

### 🔢 Mathematics (`math/`)
Mathematical utilities and algorithms.

- JIT compilation support with Numba
- Statistical functions
- Random number utilities
- Partitioning and balancing algorithms

**Use when:** You need mathematical operations, statistical computations, or performance-critical numerical code.

### 📈 Histograms (`hist/`)
Histogram creation, manipulation, and operations.

- Histogram building and merging
- Template-based histogram operations
- Integration with other data formats

**Use when:** You need to create histograms, bin data, or perform histogram-based analysis.

### ⚛️ Physics (`physics/`)
Physics-specific computations and domain logic.

- Physics object definitions
- Kinematic calculations
- Analysis-specific algorithms

**Use when:** You need physics calculations, particle object manipulations, or domain-specific physics operations.

### 📊 Plotting (`plotting/`)
Visualization and plotting utilities.

- Plot generation and styling
- Integration with matplotlib/other backends
- Analysis-specific plotting helpers

**Use when:** You need to create plots, visualize data, or generate figures for analysis.

### 🔍 Event Selection (`skimmer/`)
Event selection and filtering pipeline.

- Event filtering logic
- Selection criteria management
- Pipeline orchestration

**Use when:** You need to skim events, apply selection cuts, or build analysis pipelines.

### 🛠️ Scripts (`scripts/`)
Command-line utilities and automation tools.

- **`common.sh`** - Common shell utilities
- **`memory/`** - Memory testing and monitoring tools for preventing system hangs
  - `memory_limited_run.sh` - Universal memory guard for any command
  - `memory_monitor.py` - Python-specific memory monitoring
  - `profile_memory_simple.py` - Memory usage profiling
  - `debug_memory.py` - Advanced memory debugging tools
- Other automation and maintenance scripts

**Use when:** You need command-line tools, automation scripts, utilities to fix common problems, or memory monitoring to prevent system crashes.

### 🧰 Utilities (`utils/`)
Generic, reusable utilities that don't fit into specific domains.

- Argument parsing helpers
- JSON encoding/decoding
- String manipulation
- Retry mechanisms
- Generic wrapper functions
- **`fix_eos.py`** - Utility to repair corrupted EOS files


**Use when:** You need generic helper functions, argument parsing, or common utility operations.

### Tools for Physics Data (`tools/`)

Useful scripts and utilities for handling physics data.

 - **merge_yaml_datasets.py** - Merge multiple YAML dataset files into one
 - **merge_coffea_files.py** - Merge multiple Coffea output files


### 🧪 Tests (`tests/`)
Test suite mirroring the package structure.

- Unit tests for all modules
- Integration tests
- Test utilities and fixtures

### Friendtrees (`friendtrees/`)

Utilities to create and manage friend trees.



## Design Principles

### 1. **Domain-Driven Organization**
Each directory represents a clear conceptual domain that users can easily understand and discover.

### 2. **Dependency Direction**
Dependencies generally flow "upward" in this order:
```
utils → (storage, data_formats, math) → (hist, physics, dask) → (plotting, skimmer) → scripts
```

### 3. **User-Centric Naming**
Directory names reflect what users are trying to accomplish, not technical implementation details.

### 4. **Independent Base Components**
Components in this library should be independent of higher-level analysis code and reusable across different physics analyses.

## Common Usage Patterns

### Reading ROOT Files
```python
from src.data_formats.root import TreeReader, Friend
from src.storage import EOS

# Read from EOS
reader = TreeReader(EOS("root://path/to/file.root"))
data = reader.read()

# Work with friend trees
friend = Friend(reader, "friend_name")
```

### Array Format Conversion
```python
from src.data_formats.awkward import to_numpy
from src.data_formats.numpy import from_

# Convert awkward to numpy
np_data = to_numpy(ak_array)

# Convert from base64 numpy
array = from_.base64(encoded_data)
```

### Configuration Management
```python
from src.config import ConfigManager, config

# Load and manage configuration
config_mgr = ConfigManager("config.yaml")
my_setting = config("analysis.cuts.pt_min")
```

### Distributed Computing
```python
from src.dask_tools import delayed
from src.hist_tools_tools import hist

# Delayed histogram computation
@delayed
def compute_histogram(data):
    return hist(data, bins=50)
```

## Physics Corrections & Analysis Data (`data/`)
Stores physics corrections, calibration files, and other analysis resources (previously in `python/data/`, now in `python/base_class/data/`).

- **`Btag/`** - b-tagging scale factors and calibration files
- **`JEC/`** - jet energy corrections and related resources
- **`Muon/`** - muon corrections and scale factors
- **`PU/`** - pileup weights and corrections
- **`goldenJSON/`** - certified JSON files for luminosity sections
- **`puId/`** - pileup identification files
- ...other physics analysis resources

**Use when:** You need physics corrections, calibration files, or certified JSONs for analysis workflows.

> **Note:** All references to these files should now use the path `base_class/data/` instead of the legacy `data/` location.

## Contributing

When adding new functionality:

1. **Choose the right domain** - Place code in the directory that matches its primary purpose
2. **Keep dependencies minimal** - Avoid circular dependencies between domains
3. **Document the purpose** - Update this README when adding new directories or major functionality
4. **Add tests** - Mirror the structure in the `tests/` directory
5. **Consider users** - Think about discoverability and intuitive organization

## Future Evolution

This structure is designed to grow with the analysis needs while maintaining clear separation of concerns. New domains can be added as needed, and existing domains can be subdivided if they become too large or complex.
