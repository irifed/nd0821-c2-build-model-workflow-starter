#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact

Command line parameters:

input_artifact: name of the input artifact (dataset)
output_artifact: name of the output artifact (dataset)
output_type: type of the output artifact
output_description: description of the output artifact
min_price: parameter for cleaning (removing price outliers) - min price
max_price: parameter for cleaning (removing price outliers) - max price

Run (from the root folder):
mlflow run . -P steps=basic_cleaning
"""
import argparse
import logging

import pandas as pd

import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Downloading and reading test artifact")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info("Removing price outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting review dates to datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Removing lat/lon outliers")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned dataframe as output artifact")
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="output artifact type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="output artifact description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="min price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="max price",
        required=True
    )

    args = parser.parse_args()

    go(args)
