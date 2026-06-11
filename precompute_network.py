# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "numpy",
#     "pandas",
# ]
# ///
"""
Pre-compute the hydrofabric network graph from the flowpaths parquet file
and upload as a compact JSON for client-side upstream traversal.

Output format:
{
  "edges": [[from_id, to_id], ...],  // numeric IDs only
  "meta": { "total_nodes": N, "total_edges": N }
}

The client reconstructs the full ID prefixes:
  - nex-{id} always flows to wb-{id}
  - wb-{id} flows to nex-{to_id} (from the edges list)
  - cat-{id} = divide for wb-{id}

Usage:
    uv run precompute_network.py
    uv run precompute_network.py --skip-upload
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import List, Set

import duckdb
import numpy as np
import pandas as pd
import pickle

S3_BUCKET = "s3://communityhydrofabric/hydrofabrics/community/parquet"
PARQUET_URL = f"{S3_BUCKET}/network.parquet"
OUTPUT_FILE = "network_graph.json"

DF_CACHE_FILE = Path("dist/network_df_cache.pkl")
DF_CACHE_FILE.parent.mkdir(exist_ok=True, parents=True)

def parse_id(id_str: str):
    try:
        parts = id_str.rsplit("-", 1)
        prefix = parts[0]
        num_part = parts[1]
        if 'e' in num_part:
            # Handle scientific notation like "1e6"
            num = int(float(num_part))
        else:
            num = int(num_part)
        return prefix, num
    except Exception as e:
        raise ValueError(f"Invalid ID format: {id_str}") from e

def build_graph(use_local: str | None = None):
    """Extract id->toid edges from the network table, strip prefixes to ints."""
    use_con = False
    if DF_CACHE_FILE.exists() and not use_local:
        print(f"Loading cached DataFrame from {DF_CACHE_FILE} ...")
        with open(DF_CACHE_FILE, "rb") as f:
            df = pickle.load(f)
        print(f"  Loaded {len(df)} edges from cache")
    else:
        use_con = True
        con = duckdb.connect()

        if use_local:
            source = f"'{use_local}'"
            con.execute("INSTALL spatial; LOAD spatial;")
        else:
            con.execute("INSTALL httpfs; LOAD httpfs;")
            con.execute("SET s3_region = 'us-west-2';")
            source = f"read_parquet('{PARQUET_URL}')"

        print("Querying network for id, toid ...")
        t0 = time.perf_counter()

        # network table has id and toid columns with prefixes like wb-123, nex-456, cat-789
        df = con.sql(f"""
            SELECT id, toid
            FROM {source}
            WHERE id IS NOT NULL AND toid IS NOT NULL
        """).fetchdf()

        elapsed = time.perf_counter() - t0
        print(f"  Got {len(df)} edges in {elapsed:.1f}s")
        # Cache the DataFrame for faster subsequent runs (especially if using S3 source)
        if not use_local:
            with open(DF_CACHE_FILE, "wb") as f:
                pickle.dump(df, f)
            print(f"  Cached DataFrame to {DF_CACHE_FILE}")
    # Build compact edge list: strip prefixes, store as [from_numeric, to_numeric, from_type, to_type]
    # But actually the client needs the prefix info to know the types.
    # Let's store edges with minimal encoding:
    # type codes: 0=wb, 1=nex, 2=cat, 3=cnx, 4=tnx
    type_map = {"wb": 0, "nex": 1, "cat": 2, "cnx": 3, "tnx": 4}

    # def parse_id(id_str):
    #     try:
    #         parts = id_str.rsplit("-", 1)
    #         prefix = parts[0]
    #         num = int(parts[1])
    #         return type_map.get(prefix, -1), num
    #     except Exception as e:
    #         raise ValueError(f"Invalid ID format: {id_str}") from e

    edges = []
    skipped = 0
    for _, row in df.iterrows():
        ft, fn = parse_id(row["id"])
        tt, tn = parse_id(row["toid"])
        if ft == -1 or tt == -1:
            skipped += 1
            continue
        edges.append([ft, fn, tt, tn])

    print(f"Parsed {len(edges)} edges with known prefixes")
    
    if skipped:
        print(f"  Skipped {skipped} edges with unknown prefixes")

    # Collect unique node IDs for metadata
    nodes = set()
    for e in edges:
        nodes.add((e[0], e[1]))
        nodes.add((e[2], e[3]))
        
    print(f"Identified {len(nodes)} unique nodes in the graph")

    graph = {
        "type_codes": {v: k for k, v in type_map.items()},
        "edges": edges,
        "meta": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        },
    }
    if use_con:
        con.close()
    print(f"Finished building graph with {len(edges)} edges and {len(nodes)} unique nodes.")
    return graph


def main():
    parser = argparse.ArgumentParser(description="Pre-compute hydrofabric network graph")
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument(
        "--local-gpkg",
        type=str,
        default=None,
        help="Use local gpkg instead of S3 parquet (e.g. path/to/conus_nextgen.gpkg)",
    )
    parser.add_argument(
        "--local-parquet", type=str, default=None, help="Use local parquet file instead of S3"
    )
    args = parser.parse_args()

    source = None
    if args.local_gpkg:
        source = args.local_gpkg
    elif args.local_parquet:
        source = args.local_parquet

    graph = build_graph(use_local=source)

    out = Path(OUTPUT_FILE)
    with open(out, "w") as f:
        json.dump(graph, f, separators=(",", ":"))

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"\nWrote {out} ({size_mb:.1f} MB)")
    print(f"  {graph['meta']['total_nodes']} nodes, {graph['meta']['total_edges']} edges")

    if not args.skip_upload:
        dest = f"{S3_BUCKET}/{OUTPUT_FILE}"
        print(f"Uploading -> {dest}")
        subprocess.run(
            ["aws", "s3", "cp", str(out), dest, "--content-type", "application/json"], check=True
        )
        print("Done.")


if __name__ == "__main__":
    main()
