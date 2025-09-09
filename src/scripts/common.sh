#!/bin/bash

# Common functions and utilities for bbww analysis scripts

# Function to setup proxy if needed
setup_proxy() {
    echo "############### Including proxy"
    if [ ! -f "${PWD}/proxy/x509_proxy" ]; then
        echo "Error: x509_proxy file not found!"
        echo "Run manually:"
        echo "mkdir -p proxy && voms-proxy-init -voms cms -valid 192:00 -out ./proxy/x509_proxy"
        echo "and try again."
        exit 1
    fi
    export X509_USER_PROXY=${PWD}/proxy/x509_proxy
    echo "############### Checking proxy"
    voms-proxy-info
}

# Function to create output directory
create_output_directory() {
    local output_dir="$1"
    echo "############### Checking and creating output directory"
    echo "Output directory: $output_dir"
    if [ ! -d "$output_dir" ]; then
        mkdir -p "$output_dir"
    fi
}

# Function to display a standard section header
display_section_header() {
    local section_name="$1"
    echo -e "############### $section_name"
}

# Function to check if a file exists and is readable
check_file_exists() {
    local file_path="$1"
    local file_description="$2"
    
    if [[ ! -f "$file_path" ]]; then
        echo "Error: $file_description file not found: $file_path"
        return 1
    fi
    
    if [[ ! -r "$file_path" ]]; then
        echo "Error: $file_description file not readable: $file_path"
        return 1
    fi
    
    return 0
}

# Function to parse --output-base argument (common pattern for wrapper scripts)
parse_output_base_arg() {
    local default_output_base="${1:-output/}"
    local output_base="$default_output_base"
    
    # Parse command line arguments
    while [[ $# -gt 1 ]]; do
        case $2 in
            --output-base)
                output_base="$3"
                shift 2
                ;;
            *)
                echo "Unknown option: $2"
                echo "Usage: $0 [--output-base DIR]"
                return 1
                ;;
        esac
        shift
    done
    
    echo "$output_base"
}

# Function to validate required variables are set
validate_required_vars() {
    local vars=("$@")
    local missing_vars=()
    
    for var in "${vars[@]}"; do
        if [[ -z "${!var}" ]]; then
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        echo "Error: Missing required variables: ${missing_vars[*]}"
        return 1
    fi
    
    return 0
}

# Function to save variables to associative array (common pattern)
save_variables() {
    local -n save_array=$1
    shift
    local vars=("$@")
    
    for var in "${vars[@]}"; do
        save_array["$var"]="${!var}"
    done
}

# Function to restore variables from associative array
restore_variables() {
    local -n restore_array=$1
    shift
    local vars=("$@")
    
    for var in "${vars[@]}"; do
        declare -g "$var"="${restore_array[$var]}"
    done
}

# Function to display status message with consistent formatting
status_message() {
    local status="$1"
    local message="$2"
    echo "############### $status: $message"
}

# Function to run command with error checking
run_command() {
    local cmd=("$@")
    echo "Running: ${cmd[@]}"
    "${cmd[@]}"
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        echo "Error: Command failed with exit code $exit_code"
        return $exit_code
    fi
    
    return 0
}