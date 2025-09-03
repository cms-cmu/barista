# Documentation Generation

This directory contains the automated documentation generation system for the  project.

## Files

- `generate_docs.sh` - Main script that automatically generates documentation from README.md files
- `serve.sh` - Script to serve the documentation locally using Docker
- `mkdocs.yml` - MkDocs configuration file
- Various documentation categories:
  - `bbbb/` - HH4b Analysis documentation (generated from `python/` directory)
  - `bbWW/` - bbWW Analysis documentation (generated from `bbww/` directory)
  - `software/` - Software documentation (generated from `software/` directory)
  - `src/` - Source Code documentation (generated from `src/` directory)
  - `classifier/` - Classifier documentation (manually maintained)

## Usage

### Generating Documentation

To automatically generate documentation from README.md files throughout the repository:

```bash
cd docs/
./generate_docs.sh
```

This script will:
1. Find all README.md files in the specified directories (`python/`, `bbww/`, `software/`, `src/`)
2. Copy them to the appropriate documentation directories with proper naming
3. Skip empty README.md files
4. Update the `mkdocs.yml` navigation structure automatically
5. Create a backup of the previous `mkdocs.yml` as `mkdocs.yml.backup`

### Directory Mapping

The script maps source directories to documentation categories as follows:

| Source Directory | Documentation Category | Display Name |
|------------------|----------------------|---------------|
| `python/` | `bbbb/` | HH4b Analysis |
| `bbww/` | `bbWW/` | bbWW Analysis |
| `software/` | `software/` | Software |
| `src/` | `src/` | Source Code |

### File Naming Convention

README.md files are renamed according to their directory structure:
- `python/README.md` → `bbbb/index.md` (Overview)
- `python/plots/README.md` → `bbbb/plots.md` (Plots)
- `python/analysis/tools/README.md` → `bbbb/analysis-tools.md` (Analysis Tools)

### Serving Documentation Locally

To build and serve the documentation locally:

```bash
cd docs/
./serve.sh
```

This will start a Docker container with MkDocs and serve the documentation at:
- http://127.0.0.1:8000/ (local access)
- http://0.0.0.0:8000/ (container access)

The server will automatically reload when files change.

### Manual Documentation

The `classifier/` directory contains manually maintained documentation that is preserved during automatic generation. You can add or edit files in this directory without them being overwritten.

## Troubleshooting

### Empty README Files
The script automatically skips empty README.md files to prevent MkDocs build errors.

### Missing Navigation
If navigation items are missing after running the script, check that:
1. The source README.md files exist and are not empty
2. The source directories are accessible
3. No errors were reported during the script execution

### Build Errors
If MkDocs fails to build:
1. Check the console output for specific error messages
2. Ensure all referenced files exist
3. Verify that the `mkdocs.yml` file is valid YAML
4. Restore from backup if needed: `cp mkdocs.yml.backup mkdocs.yml`

## Customization

To modify the directory mapping or add new categories, edit the `DIRECTORY_MAPPING` and `DOC_CATEGORIES` arrays in `generate_docs.sh`.
