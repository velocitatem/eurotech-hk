"""Fetch all objects from R2 and mirror them locally under ml/data/raw/r2/."""
import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


def _client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def list_objects(s3, bucket: str) -> list[dict]:
    paginator = s3.get_paginator("list_objects_v2")
    return [obj for page in paginator.paginate(Bucket=bucket) for obj in page.get("Contents", [])]


def fetch_all(dest: Path, dry_run: bool = False, prefix: str = "") -> None:
    s3 = _client()
    bucket = os.environ["R2_BUCKET_NAME"]
    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket, **({"Prefix": prefix} if prefix else {})}

    total, skipped, downloaded = 0, 0, 0
    for page in paginator.paginate(**kwargs):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], obj["Size"]
            local = dest / key
            total += 1

            if local.exists() and local.stat().st_size == size:
                skipped += 1
                continue

            print(f"{'[dry]' if dry_run else ''} {key} ({size:,} bytes)")
            if not dry_run:
                local.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, key, str(local))
            downloaded += 1

    print(f"\n{total} objects | {downloaded} downloaded | {skipped} skipped (size match)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Mirror R2 bucket to local disk")
    p.add_argument("--dest", default="ml/data/raw/r2", help="local destination directory")
    p.add_argument("--prefix", default="", help="only fetch keys with this prefix")
    p.add_argument("--dry-run", action="store_true", help="list without downloading")
    args = p.parse_args()
    fetch_all(Path(args.dest), dry_run=args.dry_run, prefix=args.prefix)
