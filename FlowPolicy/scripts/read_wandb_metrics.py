#!/usr/bin/env python3
"""Read success_rate / loss metrics from an offline wandb run file."""

import argparse
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore
from wandb.sdk.lib import proto_util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wandb_file", type=str)
    args = parser.parse_args()

    ds = datastore.DataStore()
    ds.open_for_scan(str(Path(args.wandb_file).resolve()))
    rows = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.HasField("history"):
            hist = proto_util.dict_from_proto_list(rec.history.item)
            keep = {
                key: hist[key]
                for key in hist
                if key
                in (
                    "success_rate",
                    "train_loss",
                    "val_loss",
                    "epoch",
                    "success_rate_k1",
                    "success_rate_k2",
                    "success_rate_k3",
                    "success_rate_k4",
                    "global_step",
                )
            }
            if keep:
                rows.append(keep)

    sr_rows = [row for row in rows if "success_rate" in row]
    print(f"success_rate checkpoints: {len(sr_rows)}")
    for row in sr_rows:
        print(row)

    loss_rows = [row for row in rows if "train_loss" in row and "success_rate" not in row]
    print("--- last train/val ---")
    for row in loss_rows[-3:]:
        print(row)


if __name__ == "__main__":
    main()
