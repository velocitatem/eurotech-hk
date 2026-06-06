import os
from pathlib import Path
from typing import Protocol


class Storage(Protocol):
    def put(self, key: str, content: str | bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class LocalStorage:
    def __init__(self, base: Path = Path("ml/data/processed")):
        self.base = base

    def put(self, key: str, content: str | bytes) -> None:
        p = self.base / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content.encode() if isinstance(content, str) else content)

    def get(self, key: str) -> bytes:
        return (self.base / key).read_bytes()

    def exists(self, key: str) -> bool:
        return (self.base / key).exists()


class R2Storage:
    def __init__(self) -> None:
        import boto3
        self._s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
        self._bucket = os.environ["R2_BUCKET_NAME"]

    def put(self, key: str, content: str | bytes) -> None:
        self._s3.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=content.encode() if isinstance(content, str) else content,
        )

    def get(self, key: str) -> bytes:
        return self._s3.get_object(Bucket=self._bucket, Key=key)["Body"].read()

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False


class _MultiStorage:
    def __init__(self, *stores: Storage) -> None:
        self._stores = stores

    def put(self, key: str, content: str | bytes) -> None:
        for s in self._stores:
            s.put(key, content)

    def get(self, key: str) -> bytes:
        for s in self._stores:
            if s.exists(key):
                return s.get(key)
        raise FileNotFoundError(key)

    def exists(self, key: str) -> bool:
        return any(s.exists(key) for s in self._stores)


def make_storage(
    backend: str, local_base: Path = Path("ml/data/processed")
) -> Storage:
    match backend:
        case "local":
            return LocalStorage(local_base)
        case "r2":
            return R2Storage()
        case "both":
            return _MultiStorage(LocalStorage(local_base), R2Storage())  # type: ignore[return-value]
        case _:
            raise ValueError(f"unknown backend: {backend!r}")
