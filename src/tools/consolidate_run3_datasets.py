import os
import sys

# Configure 93TB scratch partition for local temporary file creation
scratch_tmp = "/uscmst1b_scratch/lpc1/3DayLifetime/algomez/tmp"
if os.path.exists("/uscmst1b_scratch/lpc1/3DayLifetime/algomez"):
    os.makedirs(scratch_tmp, exist_ok=True)
    os.environ["TMPDIR"] = scratch_tmp
    os.environ["_CONDOR_SCRATCH_DIR"] = scratch_tmp

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, "/srv")
import time
import gc
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
from rich.logging import RichHandler
from rich.pretty import pretty_repr
import subprocess

from src.storage.eos import EOS
from src.data_formats.root import Chunk
from src.data_formats.root.io import TreeReader, TreeWriter
from src.data_formats.awkward.zip import NanoAOD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

def check_single_chunk_on_eos(output_chunk_path: str, expected_entries: int) -> dict | None:
    """Check if a specific chunk ROOT file already exists on EOS and matches expected entries."""
    try:
        c = Chunk(output_chunk_path, fetch=True)
        if c.num_entries == expected_entries:
            logging.info(f"Chunk {os.path.basename(output_chunk_path)} already exists on EOS with {c.num_entries:,} entries. Skipping.")
            return {
                "path": str(c.path),
                "num_entries": c.num_entries,
                "uuid": str(c.uuid),
                "expected_entries": expected_entries,
                "elapsed": 0.0
            }
    except Exception:
        pass
    return None

def merge_single_chunk(output_path: str, sources: list[Chunk], step: int = 50_000, max_retries: int = 3) -> dict:
    """Merge a single partition of source chunks into one output ROOT file with retries."""
    expected_entries = sum(c.entry_stop - c.entry_start for c in sources)

    # Check if this specific chunk is already valid on EOS
    existing = check_single_chunk_on_eos(output_path, expected_entries)
    if existing is not None:
        return existing

    t0 = time.time()
    transform = NanoAOD(regular=False, jagged=True)

    last_exc = None
    for attempt in range(max_retries):
        try:
            with TreeWriter()(output_path) as writer:
                for data in TreeReader(transform=transform).iterate(*sources, step=step):
                    writer.extend(data)

            merged_chunk = writer.tree
            if isinstance(merged_chunk, list):
                merged_chunk = merged_chunk[0]

            elapsed = time.time() - t0
            logging.info(f"Finished {os.path.basename(output_path)}: {merged_chunk.num_entries:,} entries in {elapsed:.1f}s")
            gc.collect()

            return {
                "path": str(merged_chunk.path),
                "num_entries": merged_chunk.num_entries,
                "uuid": str(merged_chunk.uuid),
                "expected_entries": expected_entries,
                "elapsed": elapsed
            }
        except Exception as e:
            last_exc = e
            logging.warning(f"Attempt {attempt + 1}/{max_retries} failed for {output_path}: {e}. Retrying...")
            time.sleep(5)
            gc.collect()

    raise last_exc

def consolidate_dataset_era(
    dataset_name: str,
    year: str,
    era: str | None,
    input_files: list[str],
    output_dir_name: str,
    base_eos_out: str,
    chunk_size: int = 500_000,
    step: int = 50_000,
    max_workers: int = 4
) -> dict:
    """Consolidate all files for a dataset/era into chunk_size chunks."""
    label = f"{dataset_name} ({year} era {era})" if era else f"{dataset_name} ({year})"
    logging.info(f"=== Starting consolidation for {label} ({len(input_files)} input files) ===")

    # 1. Fetch metadata for input chunks
    t_start = time.time()
    chunks = [Chunk(f, fetch=True) for f in input_files]
    total_input_entries = sum(c.num_entries for c in chunks)
    logging.info(f"[{label}] Fetched metadata: {total_input_entries:,} total entries")

    output_dir = f"{base_eos_out.rstrip('/')}/{output_dir_name}"

    # 2. Partition into chunks
    partitions = list(Chunk.partition(chunk_size, *chunks, common_branches=True))
    n_partitions = len(partitions)
    logging.info(f"[{label}] Partitioned into {n_partitions} target chunks of ~{chunk_size:,} events")

    output_files_info = [None] * n_partitions
    to_submit = []

    # Check which chunks already exist
    for idx, part_sources in enumerate(partitions):
        chunk_file_path = f"{output_dir}/picoAOD.chunk{idx}.root"
        expected_entries = sum(c.entry_stop - c.entry_start for c in part_sources)
        existing = check_single_chunk_on_eos(chunk_file_path, expected_entries)
        if existing is not None:
            output_files_info[idx] = existing
        else:
            to_submit.append((idx, chunk_file_path, part_sources))

    if to_submit:
        logging.info(f"[{label}] Submitting {len(to_submit)}/{n_partitions} remaining chunks across {max_workers} workers...")
        with ProcessPoolExecutor(max_workers=min(max_workers, len(to_submit))) as executor:
            future_to_idx = {
                executor.submit(merge_single_chunk, chunk_file_path, part_sources, step=step): idx
                for idx, chunk_file_path, part_sources in to_submit
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result()
                    output_files_info[idx] = res
                except Exception as e:
                    logging.error(f"[{label}] Error merging chunk {idx}: {e}", exc_info=True)
                    raise e
    else:
        logging.info(f"[{label}] All {n_partitions} chunks already exist on EOS! Skipping.")

    total_merged_entries = sum(f["num_entries"] for f in output_files_info)
    elapsed_total = time.time() - t_start
    logging.info(f"[{label}] Complete in {elapsed_total:.1f}s. Merged {total_merged_entries:,} / {total_input_entries:,} entries (Match: {total_merged_entries == total_input_entries})")

    if total_merged_entries != total_input_entries:
        raise RuntimeError(f"Event count mismatch for {label}: expected {total_input_entries}, got {total_merged_entries}")

    return {
        "files": [f["path"] for f in output_files_info],
        "saved_events": total_merged_entries,
        "elapsed": elapsed_total,
        "n_files_in": len(input_files),
        "n_files_out": len(output_files_info)
    }

def main():
    parser = argparse.ArgumentParser(description="Consolidate Run 3 datasets into 500k event chunks")
    parser.add_argument("--datasets", nargs="+", default=["data", "TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"], help="Datasets to process")
    parser.add_argument("--years", nargs="+", default=["2022_preEE", "2022_EE", "2023_preBPix", "2023_BPix"], help="Years/eras to process")
    parser.add_argument("--output-base", default="root://cmseos.fnal.gov//store/user/algomez/XX4b/Run3_nanov12/", help="Base EOS directory")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Target chunk size in events")
    parser.add_argument("--step", type=int, default=50_000, help="Read step size in events")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes")
    parser.add_argument("--update-yml", action="store_true", default=True, help="Update dataset YAML manifests with new file paths")
    args = parser.parse_args()

    data_manifest_path = "coffea4bees/metadata/datasets/data.yml"
    tt_manifest_path = "coffea4bees/metadata/datasets/TT.yml"

    with open(data_manifest_path, "r") as f:
        data_manifest = yaml.safe_load(f)
    with open(tt_manifest_path, "r") as f:
        tt_manifest = yaml.safe_load(f)

    # 1. Process Collision Data
    if "data" in args.datasets:
        logging.info("================ PROCESSING COLLISION DATA ================")
        for year in args.years:
            if year not in data_manifest.get("data", {}):
                continue
            if "picoAOD" not in data_manifest["data"][year]:
                continue

            for era, era_info in data_manifest["data"][year]["picoAOD"].items():
                input_files = era_info.get("files", [])
                if not input_files:
                    continue

                if year == "2022_EE":
                    dir_name = f"data_2022_EE{era}"
                elif year == "2022_preEE":
                    dir_name = f"data_2022_preEE{era}"
                elif year == "2023_BPix":
                    dir_name = f"data_2023_BPix{era}"
                elif year == "2023_preBPix":
                    dir_name = f"data_2023_preBPix{era}"
                else:
                    dir_name = f"data_{year}_{era}"

                logging.info(f"\n--- Data {year} Era {era} -> {dir_name} ---")
                res = consolidate_dataset_era(
                    dataset_name="data",
                    year=year,
                    era=era,
                    input_files=input_files,
                    output_dir_name=dir_name,
                    base_eos_out=args.output_base,
                    chunk_size=args.chunk_size,
                    step=args.step,
                    max_workers=args.workers
                )

                if args.update_yml:
                    data_manifest["data"][year]["picoAOD"][era]["files"] = res["files"]
                    data_manifest["data"][year]["picoAOD"][era]["saved_events"] = res["saved_events"]
                    with open(data_manifest_path, "w") as f:
                        yaml.dump(data_manifest, f, default_flow_style=False, sort_keys=False)

    # 2. Process TT Samples
    tt_samples = [d for d in args.datasets if d in ["TTTo2L2Nu", "TTToHadronic", "TTToSemiLeptonic"]]
    if tt_samples:
        logging.info("================ PROCESSING TTBAR SAMPLES ================")
        for sample in tt_samples:
            if sample not in tt_manifest:
                continue

            for year in args.years:
                if year not in tt_manifest[sample]:
                    continue
                if "picoAOD" not in tt_manifest[sample][year]:
                    continue

                sample_info = tt_manifest[sample][year]["picoAOD"]
                input_files = sample_info.get("files", [])
                if not input_files:
                    continue

                dir_name = f"{sample}_{year}"
                logging.info(f"\n--- {sample} {year} -> {dir_name} ---")
                res = consolidate_dataset_era(
                    dataset_name=sample,
                    year=year,
                    era=None,
                    input_files=input_files,
                    output_dir_name=dir_name,
                    base_eos_out=args.output_base,
                    chunk_size=args.chunk_size,
                    step=args.step,
                    max_workers=args.workers
                )

                if args.update_yml:
                    tt_manifest[sample][year]["picoAOD"]["files"] = res["files"]
                    tt_manifest[sample][year]["picoAOD"]["saved_events"] = res["saved_events"]
                    with open(tt_manifest_path, "w") as f:
                        yaml.dump(tt_manifest, f, default_flow_style=False, sort_keys=False)

    logging.info("All consolidations completed successfully!")

if __name__ == "__main__":
    main()
