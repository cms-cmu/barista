"""
Merge parent picoAOD file (keeping run, luminosityBlock, event, and Jet collection)
with all its friendtrees into single Parquet files.
Supports three output formats:
- padded (Option 1): expand collections to fixed-length flat columns (e.g. Jet1_pt, Jet2_pt, ...).
- relational (Option 2): save separate event, jet, can_jet, and not_can_jet tables.
- list (Option 3): save columns as variable-length lists at the top level.

Groups manifests by base dataset and year/era, and appends a 'year' string column.
Includes memory optimizations (garbage collection, smaller default chunks) for LPC memory watchdogs.
Gracefully skips and cleans up corrupted/mismatched parent files.
"""

import os
import re
import glob
import json
import logging
import argparse
import gc
import numpy as np
import awkward as ak
import uproot
import pyarrow.parquet as pq
import pyarrow as pa

from src.data_formats.root import Friend
from src.data_formats.root.chunk import Chunk
from src.data_formats.awkward.zip import NanoAOD
from src.friendtrees.merge_friend_meta import _iter_friend_dicts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def parse_manifest_name(filepath):
    """
    Parse the base dataset name and year/era from a manifest file path.
    E.g. all_friends_TTTo2L2Nu_UL16_postVFP.json -> base='TTTo2L2Nu', year='UL16_postVFP'
         all_friends_data_UL17_C.json -> base='data', year='UL17_C'
    """
    basename = os.path.basename(filepath)
    match = re.match(r"all_friends_(.*)_(UL\d{2}(?:_preVFP|_postVFP)?(?:_.*)?)\.json$", basename)
    if match:
        return match.group(1), match.group(2)
        
    # Fallback if it does not match standard pattern
    match_simple = re.match(r"all_friends_(.*)\.json$", basename)
    if match_simple:
        return match_simple.group(1), "unknown"
    return None, None

def get_newest_manifests(input_glob):
    """
    Search for all files matching input_glob and resolve duplicate datasets,
    keeping only the newest manifest file based on file modification time.
    """
    files = glob.glob(input_glob)
    if not files:
        logging.warning(f"No manifest files found matching glob: {input_glob}")
        return {}

    dataset_manifests = {}
    for f in files:
        base_name, year_era = parse_manifest_name(f)
        if not base_name:
            continue
        key = (base_name, year_era)
        mtime = os.path.getmtime(f)
        
        # If key is new or the current file is newer, update
        if key not in dataset_manifests or mtime > dataset_manifests[key]["mtime"]:
            dataset_manifests[key] = {
                "path": f,
                "mtime": mtime,
                "base_name": base_name,
                "year_era": year_era
            }
            
    return dataset_manifests

def load_friends_from_manifest(manifest_path):
    """
    Load friendtree metadata from the JSON manifest file and parse into Friend objects.
    """
    with open(manifest_path, "r") as f:
        meta = json.load(f)
        
    friends = {}
    for v in _iter_friend_dicts(meta):
        friend = Friend.from_json(v)
        friends[friend.name] = friend
    return friends

def get_aligned_parent_chunks(friends):
    """
    Get the parent picoAOD chunks from the parsed Friend objects.
    Since all friend collections are aligned to the same parent files,
    we can extract parent chunks from any of the parsed Friend objects.
    """
    if not friends:
        return []
    first_friend = list(friends.values())[0]
    return list(first_friend._data.keys())

def expand_collection_to_columns(array, collection_prefix, max_objects, fill_value=-1):
    """
    Pads and flattens a jagged list collection attribute into max_objects individual flat columns.
    E.g. Jet_pt (var * float) -> Jet1_pt, Jet2_pt, ... Jet12_pt.
    """
    fields = [f for f in array.fields if f.startswith(collection_prefix + "_")]
    expanded = {}
    for f in fields:
        attr_name = f[len(collection_prefix) + 1:]  # E.g. 'pt'
        padded = ak.pad_none(array[f], max_objects, axis=1, clip=True)
        
        # Assign default fill values based on standard data types
        current_fill = fill_value
        if "Id" in attr_name or "Idx" in attr_name or "Flavour" in attr_name:
            current_fill = -1
        elif "isSelJet" in attr_name or "cleanmask" in attr_name:
            current_fill = 0
            
        for idx in range(max_objects):
            col_name = f"{collection_prefix}{idx + 1}_{attr_name}"
            col_data = padded[:, idx]
            expanded[col_name] = ak.fill_none(col_data, current_fill)
    return expanded

def make_padded_array(flat_array, max_jets=12, max_can_jets=4, max_not_can_jets=8, fill_value=-1):
    """
    Convert all variable-length list collections into fixed-length padded columns.
    """
    expanded = {}
    expanded.update(expand_collection_to_columns(flat_array, "Jet", max_jets, fill_value))
    expanded.update(expand_collection_to_columns(flat_array, "CanJet", max_can_jets, fill_value))
    expanded.update(expand_collection_to_columns(flat_array, "NotCanJet", max_not_can_jets, fill_value))
    
    # Event-level columns + counts are preserved
    combined_dict = {}
    for field in flat_array.fields:
        if any(field.startswith(prefix + "_") for prefix in ("Jet", "CanJet", "NotCanJet")):
            continue
        combined_dict[field] = flat_array[field]
        
    combined_dict.update(expanded)
    return ak.Array(combined_dict)

def merge_and_save_dataset_year(base_name, year_era, manifest_path, output_dir, 
                                format_layout="padded", trial_mode=False, 
                                chunk_size=50000, zip_mode=False,
                                max_jets=12, max_can_jets=4, max_not_can_jets=8,
                                fill_value=-1):
    """
    Merge parent picoAOD file and all associated friendtrees for a given dataset and year/era.
    Filters events in SR or SB, appends a string 'year' column, and saves to Parquet in the chosen format.
    """
    friends = load_friends_from_manifest(manifest_path)
    if not friends:
        logging.error(f"No friend trees found in manifest {manifest_path}")
        return
        
    parent_chunks = get_aligned_parent_chunks(friends)
    if not parent_chunks:
        logging.error(f"No parent picoAOD chunks found in manifest {manifest_path}")
        return
        
    logging.info(f"Dataset '{base_name}' ({year_era}) has {len(parent_chunks)} parent files. Friends: {list(friends.keys())}")
    
    # Initialize NanoAOD zip transform if zip_mode is requested (only applies to 'list' layout)
    transform = NanoAOD(regular=False, jagged=True) if (zip_mode and format_layout == "list") else None
    
    # Setup base output target folder (Option 2 directory dataset)
    output_dir_base = os.path.join(output_dir, base_name)
    os.makedirs(output_dir_base, exist_ok=True)
    
    # If trial mode, only process the first parent chunk and limit events
    if trial_mode:
        parent_chunks = parent_chunks[:1]
        logging.info("Running in TRIAL mode. Only processing the first parent file.")
        
    # Global tracking of event_idx for relational format
    global_event_idx = 0
        
    for parent_idx, parent_chunk in enumerate(parent_chunks):
        parent_path = parent_chunk.path
        
        # Determine output file paths
        part_prefix = "trial_part" if trial_mode else "part"
        part_suffix = f"_{parent_idx}" if len(parent_chunks) > 1 else ""
        
        if format_layout == "relational":
            output_files = {
                "events": os.path.join(output_dir_base, f"{part_prefix}_{year_era}{part_suffix}_events.parquet"),
                "jets": os.path.join(output_dir_base, f"{part_prefix}_{year_era}{part_suffix}_jets.parquet"),
                "can_jets": os.path.join(output_dir_base, f"{part_prefix}_{year_era}{part_suffix}_can_jets.parquet"),
                "not_can_jets": os.path.join(output_dir_base, f"{part_prefix}_{year_era}{part_suffix}_not_can_jets.parquet")
            }
        else:
            output_file = os.path.join(output_dir_base, f"{part_prefix}_{year_era}{part_suffix}.parquet")
            output_files = {"single": output_file}
            
        logging.info(f"Processing parent file {parent_idx + 1}/{len(parent_chunks)}: {parent_path}")
        for k, v in output_files.items():
            logging.info(f"  Output Parquet ({k}): {v}")
        
        # Get friend items matching this parent file for each friend tree
        friend_items = {}
        for friend_name, friend in friends.items():
            friend_items[friend_name] = friend._data.get(parent_chunk, [])
            
        # Typically, there is 1 friend item covering the entry range [0, parent_entries)
        # But we loop over the items to be general
        first_friend_name = list(friends.keys())[0]
        first_items = friend_items[first_friend_name]
        
        writers = {}
        skip_parent = False
        
        for item_idx, item in enumerate(first_items):
            start = item.start
            stop = item.stop
            total_entries = stop - start
            
            logging.info(f"  Friend item {item_idx + 1}/{len(first_items)}: entry range [{start}, {stop}) ({total_entries} events)")
            
            # Sub-range streaming to control memory usage
            sub_chunk_size = chunk_size
            if trial_mode:
                # In trial mode, process at most 10,000 events
                stop = min(start + 10000, stop)
                sub_chunk_size = 10000
                total_entries = stop - start
                logging.info(f"  Trial mode: limiting to entry range [{start}, {stop}) ({total_entries} events)")
                
            for sub_start in range(start, stop, sub_chunk_size):
                sub_stop = min(sub_start + sub_chunk_size, stop)
                logging.info(f"    Streaming sub-range [{sub_start}, {sub_stop})...")
                
                # 1. Read parent picoAOD branches
                with uproot.open(parent_path) as f_parent:
                    tree_parent = f_parent["Events"]
                    all_keys = tree_parent.keys()
                    parent_branches = [
                        k for k in all_keys 
                        if k in ("run", "luminosityBlock", "event", "nJet") or k.startswith("Jet_")
                    ]
                    parent_arr = tree_parent.arrays(
                        parent_branches, 
                        entry_start=sub_start, 
                        entry_stop=sub_stop
                    )
                    
                # 2. Read corresponding entries from friend trees
                friend_sub_start = sub_start - start
                friend_sub_stop = sub_stop - start
                
                friend_sub_arrays = {}
                for f_name in friends.keys():
                    f_item = friend_items[f_name][item_idx]
                    f_path = f_item.chunk.path
                    
                    with uproot.open(f_path) as f_friend:
                        tree_friend = f_friend["Events"]
                        friend_arr = tree_friend.arrays(
                            entry_start=friend_sub_start, 
                            entry_stop=friend_sub_stop
                        )
                        # Quick validation
                        if len(friend_arr) != len(parent_arr):
                            logging.error(
                                f"Event alignment mismatch! Parent has {len(parent_arr)} events but "
                                f"friend '{f_name}' has {len(friend_arr)} events in range [{sub_start}, {sub_stop}]. "
                                "Skipping this parent file due to data mismatch/corruption."
                            )
                            skip_parent = True
                            break
                        friend_sub_arrays[f_name] = friend_arr
                        
                if skip_parent:
                    break
                    
                # 3. Combine parent and friend arrays
                combined_dict = {}
                for field in parent_arr.fields:
                    combined_dict[field] = parent_arr[field]
                for f_name, friend_arr in friend_sub_arrays.items():
                    for field in friend_arr.fields:
                        combined_dict[field] = friend_arr[field]
                        
                # Append year column as a string
                combined_dict["year"] = ak.Array([year_era] * len(parent_arr))
                
                flat_array = ak.Array(combined_dict)
                
                # 4. Apply Event Selection: Keep only events where SR | SB is True
                if "SR" in flat_array.fields and "SB" in flat_array.fields:
                    mask = (flat_array.SR != 0) | (flat_array.SB != 0)
                    filtered_array = flat_array[mask]
                else:
                    logging.warning("SR or SB flags not found in array. Skipping event selection filter.")
                    filtered_array = flat_array
                    
                n_filtered_events = len(filtered_array)
                logging.info(f"    Filtered sub-range: {n_filtered_events} events kept out of {len(flat_array)}")
                
                if n_filtered_events > 0:
                    # 5. Format output layout
                    if format_layout == "padded":
                        processed = make_padded_array(
                            filtered_array, 
                            max_jets=max_jets, 
                            max_can_jets=max_can_jets, 
                            max_not_can_jets=max_not_can_jets,
                            fill_value=fill_value
                        )
                        table = ak.to_arrow_table(processed)
                        
                        if "single" not in writers:
                            writers["single"] = pq.ParquetWriter(output_files["single"], table.schema)
                        writers["single"].write_table(table)
                        
                    elif format_layout == "relational":
                        # Generate unique event indices for linking
                        event_idx = np.arange(global_event_idx, global_event_idx + n_filtered_events)
                        
                        # A. Events Table
                        events_dict = {"event_idx": event_idx}
                        for field in filtered_array.fields:
                            if any(field.startswith(prefix + "_") for prefix in ("Jet", "CanJet", "NotCanJet")):
                                continue
                            events_dict[field] = filtered_array[field]
                        events_table = ak.Array(events_dict)
                        events_arrow = ak.to_arrow_table(events_table)
                        if "events" not in writers:
                            writers["events"] = pq.ParquetWriter(output_files["events"], events_arrow.schema)
                        writers["events"].write_table(events_arrow)
                        
                        # B. Jets Table
                        repeated_jet_idx = np.repeat(event_idx, filtered_array.nJet)
                        jets_dict = {"event_idx": repeated_jet_idx}
                        for field in filtered_array.fields:
                            if field.startswith("Jet_"):
                                jets_dict[field[4:]] = ak.flatten(filtered_array[field])
                        jets_table = ak.Array(jets_dict)
                        jets_arrow = ak.to_arrow_table(jets_table)
                        if "jets" not in writers:
                            writers["jets"] = pq.ParquetWriter(output_files["jets"], jets_arrow.schema)
                        writers["jets"].write_table(jets_arrow)
                        
                        # C. CanJets Table
                        repeated_can_idx = np.repeat(event_idx, filtered_array.nCanJet)
                        can_dict = {"event_idx": repeated_can_idx}
                        for field in filtered_array.fields:
                            if field.startswith("CanJet_"):
                                can_dict[field[7:]] = ak.flatten(filtered_array[field])
                        can_table = ak.Array(can_dict)
                        can_arrow = ak.to_arrow_table(can_table)
                        if "can_jets" not in writers:
                            writers["can_jets"] = pq.ParquetWriter(output_files["can_jets"], can_arrow.schema)
                        writers["can_jets"].write_table(can_arrow)
                        
                        # D. NotCanJets Table
                        repeated_not_can_idx = np.repeat(event_idx, filtered_array.nNotCanJet)
                        not_can_dict = {"event_idx": repeated_not_can_idx}
                        for field in filtered_array.fields:
                            if field.startswith("NotCanJet_"):
                                not_can_dict[field[10:]] = ak.flatten(filtered_array[field])
                        not_can_table = ak.Array(not_can_dict)
                        not_can_arrow = ak.to_arrow_table(not_can_table)
                        if "not_can_jets" not in writers:
                            writers["not_can_jets"] = pq.ParquetWriter(output_files["not_can_jets"], not_can_arrow.schema)
                        writers["not_can_jets"].write_table(not_can_arrow)
                        
                        # Increment global event index
                        global_event_idx += n_filtered_events
                        
                    else:  # list layout
                        if transform is not None:
                            processed = transform(filtered_array)
                        else:
                            processed = filtered_array
                        table = ak.to_arrow_table(processed)
                        
                        if "single" not in writers:
                            writers["single"] = pq.ParquetWriter(output_files["single"], table.schema)
                        writers["single"].write_table(table)
                        
                # Explicit Memory Clean-up & Garbage Collection to prevent OOM kills on interactive nodes
                del parent_arr
                for f_name in list(friend_sub_arrays.keys()):
                    del friend_sub_arrays[f_name]
                del friend_sub_arrays
                del combined_dict
                del flat_array
                del filtered_array
                if format_layout == "padded" and n_filtered_events > 0:
                    del processed
                    del table
                elif format_layout == "relational" and n_filtered_events > 0:
                    del event_idx
                    del events_dict, events_table, events_arrow
                    del repeated_jet_idx, jets_dict, jets_table, jets_arrow
                    del repeated_can_idx, can_dict, can_table, can_arrow
                    del repeated_not_can_idx, not_can_dict, not_can_table, not_can_arrow
                elif format_layout == "list" and n_filtered_events > 0:
                    del processed
                    del table
                gc.collect()
                
            if trial_mode or skip_parent:
                break
                
        # Close all active writers
        for w in list(writers.keys()):
            writers[w].close()
            del writers[w]
        del writers
        
        # If we skipped processing due to alignment mismatch, remove the incomplete output files
        if skip_parent:
            logging.warning(f"Skipping parent file {parent_path} due to alignment mismatch. Cleaning up partial outputs.")
            for k, v in output_files.items():
                if os.path.exists(v):
                    os.remove(v)
            continue
            
        logging.info(f"Saved merged dataset parts to output directory.")
        
        # Print file summary for validation
        if trial_mode and format_layout == "padded":
            out_file = output_files["single"]
            info = pq.read_metadata(out_file)
            print("\n" + "="*50)
            print("TRIAL MERGE (PADDED FORMAT) COMPLETED SUCCESSFULLY!")
            print(f"File: {out_file}")
            print(f"Number of rows (SR | SB == True): {info.num_rows}")
            print(f"File size on disk: {os.path.getsize(out_file) / 1024:.2f} KB")
            print("Schema (Partial list showing year column and expanded padded columns):")
            schema = pq.read_schema(out_file)
            for idx, field in enumerate(schema):
                if idx < 35 or field.name == "year":
                    print(f"  {field}")
            print(f"  ... and {len(schema) - 35} more fields.")
            print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Merge parent picoAOD and friend trees into Parquet format.")
    parser.add_argument(
        "-i", "--input-glob",
        default="output/mixeddata_friendtrees*/singlefiles/all_friends_*.json",
        help="Glob pattern for friend tree manifest JSONs (default: %(default)s)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="output/mixeddata_parquet",
        help="Output directory for parquet files (default: %(default)s)"
    )
    parser.add_argument(
        "-d", "--dataset",
        help="Process a specific base dataset by name (e.g. TTTo2L2Nu). If not set, processes all."
    )
    parser.add_argument(
        "-m", "--manifest",
        help="Path to a specific friend tree manifest JSON file to process (supercedes --dataset)."
    )
    parser.add_argument(
        "--format",
        choices=["padded", "relational", "list"],
        default="padded",
        help="Output layout format: padded (Option 1), relational (Option 2), or list (Option 3). (default: %(default)s)"
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run a trial merge on a single parent chunk (limiting to 10k events) of the selected dataset."
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Zip collections into nested structures. Only applies when --format is 'list'."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Chunk size for reading/writing in number of events (default: %(default)s)"
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=12,
        help="Max number of jets for padding/truncating (default: %(default)s)"
    )
    parser.add_argument(
        "--max-can-jets",
        type=int,
        default=4,
        help="Max number of candidate jets for padding/truncating (default: %(default)s)"
    )
    parser.add_argument(
        "--max-not-can-jets",
        type=int,
        default=8,
        help="Max number of non-candidate jets for padding/truncating (default: %(default)s)"
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=-1.0,
        help="Padding value for missing/truncated collection elements (default: %(default)s)"
    )

    args = parser.parse_args()
    
    # Process a single explicit manifest file if requested (HTCondor use case)
    if args.manifest:
        base_name, year_era = parse_manifest_name(args.manifest)
        if not base_name:
            logging.error(f"Could not parse base name and year from manifest path: {args.manifest}")
            return
        logging.info(f"Processing specific manifest: '{args.manifest}' (base: {base_name}, era: {year_era})")
        merge_and_save_dataset_year(
            base_name,
            year_era,
            args.manifest,
            args.output_dir,
            format_layout=args.format,
            trial_mode=args.trial,
            chunk_size=args.chunk_size,
            zip_mode=args.zip,
            max_jets=args.max_jets,
            max_can_jets=args.max_can_jets,
            max_not_can_jets=args.max_not_can_jets,
            fill_value=args.fill_value
        )
        return
        
    # Resolve all base datasets and year/eras across manifests
    newest_manifests = get_newest_manifests(args.input_glob)
    
    if not newest_manifests:
        logging.error("No valid dataset manifests discovered. Exiting.")
        return
        
    logging.info(f"Discovered {len(newest_manifests)} unique dataset-year keys in manifests.")
    
    # Group manifests by base_name
    grouped_manifests = {}
    for key, info in newest_manifests.items():
        base_name = info["base_name"]
        if base_name not in grouped_manifests:
            grouped_manifests[base_name] = []
        grouped_manifests[base_name].append(info)
        
    if args.dataset:
        if args.dataset not in grouped_manifests:
            logging.error(f"Dataset '{args.dataset}' not found in resolved manifests: {list(grouped_manifests.keys())}")
            return
        
        logging.info(f"Selected dataset: '{args.dataset}'. Processing all available years/eras: {[m['year_era'] for m in grouped_manifests[args.dataset]]}")
        
        manifests_to_run = grouped_manifests[args.dataset]
        if args.trial:
            manifests_to_run = manifests_to_run[:1]
            
        for m_info in manifests_to_run:
            merge_and_save_dataset_year(
                m_info["base_name"],
                m_info["year_era"],
                m_info["path"],
                args.output_dir,
                format_layout=args.format,
                trial_mode=args.trial,
                chunk_size=args.chunk_size,
                zip_mode=args.zip,
                max_jets=args.max_jets,
                max_can_jets=args.max_can_jets,
                max_not_can_jets=args.max_not_can_jets,
                fill_value=args.fill_value
            )
    else:
        if args.trial:
            base_name = list(grouped_manifests.keys())[0]
            m_info = grouped_manifests[base_name][0]
            logging.info(f"No dataset specified with --trial. Selecting first dataset: '{base_name}', era: '{m_info['year_era']}'")
            merge_and_save_dataset_year(
                m_info["base_name"],
                m_info["year_era"],
                m_info["path"],
                args.output_dir,
                format_layout=args.format,
                trial_mode=True,
                chunk_size=args.chunk_size,
                zip_mode=args.zip,
                max_jets=args.max_jets,
                max_can_jets=args.max_can_jets,
                max_not_can_jets=args.max_not_can_jets,
                fill_value=args.fill_value
            )
        else:
            # Process all datasets sequentially
            for base_name, manifests in grouped_manifests.items():
                logging.info(f"Processing dataset '{base_name}' across all available years...")
                for m_info in manifests:
                    try:
                        merge_and_save_dataset_year(
                            m_info["base_name"],
                            m_info["year_era"],
                            m_info["path"],
                            args.output_dir,
                            format_layout=args.format,
                            trial_mode=False,
                            chunk_size=args.chunk_size,
                            zip_mode=args.zip,
                            max_jets=args.max_jets,
                            max_can_jets=args.max_can_jets,
                            max_not_can_jets=args.max_not_can_jets,
                            fill_value=args.fill_value
                        )
                    except Exception as e:
                        logging.exception(f"Failed to process dataset '{base_name}' era '{m_info['year_era']}': {e}")

if __name__ == "__main__":
    main()
