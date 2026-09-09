"""Adapters for SEB log pipeline dependencies."""

from .s3_object_store import LocalFileStore, S3ObjectStore

__all__ = ["LocalFileStore", "S3ObjectStore"]
