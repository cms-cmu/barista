#!/usr/bin/env bash

# If sourced, just print a note and return without touching shell options
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Please run (do not source): bash ${BASH_SOURCE[0]} [options]"
  return 0 2>/dev/null || exit 0
fi

set -euo pipefail

usage() {
  echo "Usage: $0 -s <source> -d <destination> [options]"
  echo "Required options:"
  echo "  -s  Source folder (local path or root://cmseos.fnal.gov/... URL)"
  echo "      Can point to a specific folder to transfer all files from it"
  echo "  -d  Destination folder (CERNBox path)"
  echo "      If relative path, will be prefixed with /eos/user/<first_letter>/\$USER/"
  echo "Optional options:"
  echo "  -f  Folder filter regex"
  echo "  -F  File filter regex"
  echo "  -t  Tar files before transfer, then extract remotely (if /eos is mounted)"
  echo "  -j  Number of parallel transfers (default: 1)"
  echo "  -h  Show this help message"
  exit "${1:-0}"
}

# Early help handling
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage 0
fi

command -v gfal-copy >/dev/null 2>&1 || { echo "gfal-copy not found"; exit 1; }

# Default values
source_folder=""
destination_folder=""
folder_filter=""
file_filter=""
use_tar=0
jobs=1

# Parse command line arguments
while getopts s:d:f:F:tj: flag
do
    case "${flag}" in
        s) source_folder=${OPTARG};;
        d) destination_folder=${OPTARG};;
        f) folder_filter=${OPTARG};;
        F) file_filter=${OPTARG};;
        t) use_tar=1;;
        j) jobs=${OPTARG};;
    esac
done

# Check required arguments
if [[ -z $source_folder || -z $destination_folder ]]; then
  echo "Error: -s and -d options are required"
  usage 1
fi

# Convert relative destination paths to absolute CERNBox paths
if [[ ! $destination_folder =~ ^/ ]]; then
  FIRST_LETTER="${USER:0:1}"
  destination_folder="/eos/user/${FIRST_LETTER}/${USER}/${destination_folder}"
fi

# Convert relative paths to absolute paths for local sources
if [[ $source_folder =~ ^root:// ]]; then
  list_cmd="eosls"
else
  list_cmd="ls"
  source_folder="$(cd "$source_folder" && pwd)"
fi

if [[ $use_tar -eq 1 ]]; then
  echo "Packing $source_folder into a tarball for transfer..."
  if [[ ! $source_folder =~ ^root:// ]]; then
      # Need a robust temp dir
      tmp_dir="$(mktemp -d)"
      trap 'rm -rf "$tmp_dir"' EXIT
      archive_name="transfer_$(date +%s).tar.gz"
      tar -C "$source_folder" -czf "$tmp_dir/$archive_name" .
      
      echo "Transferring tarball ${archive_name}"
      src="file://${tmp_dir}/${archive_name}"
      dst="root://eosuser.cern.ch${destination_folder}/${archive_name}"
      # Provide parent directory creation via -p if supported (gfal-copy normally creates parents if specified or wait, gfal-copy creates parent if needed? Let's make sure destination exists might need -p ... gfal-copy does it automatically or requires --parents. Let's just use gfal-copy and assume structure or it creates it, the existing script doesn't use --parents but we are transferring a single file, it's safer to just run it)
      gfal-copy -f "$src" "$dst"
      
      # Optional remote unpack if eos is mounted locally
      if [[ -d "$destination_folder" ]]; then
          echo "Extracting tarball natively since $destination_folder is accessible..."
          tar -xzf "$destination_folder/$archive_name" -C "$destination_folder"
          rm -f "$destination_folder/$archive_name"
      else
          echo "NOTE: Remote CERNBox fuse mount ($destination_folder) is not locally accessible."
          echo "Tarball was transferred but not extracted."
      fi
      
      echo "Done."
      exit 0
  else
      echo "Error: -t (tar functionality) is currently only supported when source is a local path."
      exit 1
  fi
fi

# Check if source is a folder with subfolders or a direct folder with files
first_item=$($list_cmd "$source_folder" | head -1)

command_file=$(mktemp)
trap 'rm -f "$command_file"' EXIT

# If the first item is a file (has extension), treat source as direct folder
if [[ $first_item == *.* ]]; then
  # Transfer files directly from source folder
  for ifile in $($list_cmd "$source_folder");
  do
      if [[ -z $file_filter || $ifile =~ $file_filter ]];
      then
          # Build source URL for gfal-copy
          if [[ $source_folder =~ ^root:// ]]; then
            src="${source_folder}/${ifile}"
          else
            src="file://${source_folder}/${ifile}"
          fi
          
          dst="root://eosuser.cern.ch${destination_folder}/${ifile}"
          printf "%s\0%s\0" "$src" "$dst" >> "$command_file"
      fi
  done
else
  # Transfer files from subfolders
  for ifolder in $($list_cmd "$source_folder");
  do
      if [[ -z $folder_filter || $ifolder =~ $folder_filter ]];
      then
          for ifile in $($list_cmd "${source_folder}/${ifolder}");
          do
              if [[ -z $file_filter || $ifile =~ $file_filter ]];
              then
                  # Build source URL for gfal-copy
                  if [[ $source_folder =~ ^root:// ]]; then
                    src="${source_folder}/${ifolder}/${ifile}"
                  else
                    src="file://${source_folder}/${ifolder}/${ifile}"
                  fi
                  
                  dst="root://eosuser.cern.ch${destination_folder}/${ifolder}/${ifile}"
                  printf "%s\0%s\0" "$src" "$dst" >> "$command_file"
              fi
          done
      fi
  done
fi

if [[ $jobs -gt 1 ]]; then
  echo "Found $(grep -z -c . "$command_file" | awk '{print $1/2}') files to transfer."
  echo "Transfers will be run in parallel using $jobs workers..."
  xargs -0 -n 2 -P "$jobs" bash -c 'echo "Transferring: $0"; gfal-copy -f "$0" "$1" && echo "  => Done: $0" || echo "  => Failed: $0"'
else
  echo "Found $(grep -z -c . "$command_file" | awk '{print $1/2}') files to transfer sequentially..."
  xargs -0 -n 2 -P 1 bash -c 'echo "Transferring: $0"; gfal-copy -f "$0" "$1" && echo "  => Done: $0" || echo "  => Failed: $0"'
fi

echo "Done."