"""
S3/MinIO Storage Service for FlowML

This service provides abstracted storage operations that work with both:
- Local filesystem (development mode)
- MinIO/S3 (production mode)

Usage:
    from services.storage import storage_service
    
    # Upload a file
    uri = await storage_service.upload_file("datasets", local_path, "myfile.csv")
    
    # Download a file  
    await storage_service.download_file(uri, local_path)
    
    # Get presigned URL for direct access
    url = await storage_service.get_presigned_url(uri)
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

# Check if boto3 is available
try:
    import boto3
    from botocore.exceptions import ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not installed, S3 storage unavailable. Install with: pip install boto3")


class StorageService:
    """
    Unified storage interface supporting local filesystem and S3/MinIO.
    """
    
    def __init__(self):
        self.mode = os.getenv("STORAGE_MODE", "local")  # local or s3
        self.local_base = Path(os.getenv("LOCAL_STORAGE_PATH", "./uploads"))
        
        # S3/MinIO config
        self.s3_endpoint = os.getenv("S3_ENDPOINT", "http://localhost:9000")
        self.s3_access_key = os.getenv("S3_ACCESS_KEY", "flowml-admin")
        self.s3_secret_key = os.getenv("S3_SECRET_KEY", "flowml-secret")
        self.s3_region = os.getenv("S3_REGION", "us-east-1")
        self.s3_bucket_prefix = os.getenv("S3_BUCKET_PREFIX", "flowml")
        
        # Initialize S3 client if in s3 mode
        self._s3_client = None
        if self.mode == "s3" and BOTO3_AVAILABLE:
            self._init_s3_client()
    
    def _init_s3_client(self):
        """Initialize boto3 S3 client for MinIO"""
        try:
            self._s3_client = boto3.client(
                's3',
                endpoint_url=self.s3_endpoint,
                aws_access_key_id=self.s3_access_key,
                aws_secret_access_key=self.s3_secret_key,
                region_name=self.s3_region,
            )
            logger.info(f"S3 client initialized for endpoint: {self.s3_endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self._s3_client = None
    
    @property
    def is_s3_mode(self) -> bool:
        """Check if running in S3 mode with valid client"""
        return self.mode == "s3" and self._s3_client is not None
    
    def _get_bucket_name(self, bucket_type: str) -> str:
        """Get full bucket name with prefix"""
        return f"{self.s3_bucket_prefix}-{bucket_type}"
    
    def _ensure_local_dir(self, bucket_type: str) -> Path:
        """Ensure local directory exists for a bucket type"""
        path = self.local_base / bucket_type
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    async def upload_file(
        self,
        bucket_type: str,  # datasets, artifacts, runs, logs
        file_path: Union[str, Path],
        object_name: Optional[str] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file to storage.
        
        Args:
            bucket_type: Type of storage bucket (datasets, artifacts, runs, logs)
            file_path: Local path to the file to upload
            object_name: Name to store the file as (defaults to filename)
            content_type: MIME type of the file
            
        Returns:
            URI string: s3://bucket/key or file://path
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        object_name = object_name or file_path.name
        
        if self.is_s3_mode:
            return await self._upload_s3(bucket_type, file_path, object_name, content_type)
        else:
            return await self._upload_local(bucket_type, file_path, object_name)
    
    async def _upload_s3(
        self,
        bucket_type: str,
        file_path: Path,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """Upload file to S3/MinIO"""
        bucket_name = self._get_bucket_name(bucket_type)
        
        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type
        
        try:
            self._s3_client.upload_file(
                str(file_path),
                bucket_name,
                object_name,
                ExtraArgs=extra_args if extra_args else None
            )
            uri = f"s3://{bucket_name}/{object_name}"
            logger.info(f"Uploaded to S3: {uri}")
            return uri
        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    async def _upload_local(
        self,
        bucket_type: str,
        file_path: Path,
        object_name: str
    ) -> str:
        """Copy file to local storage directory"""
        dest_dir = self._ensure_local_dir(bucket_type)
        dest_path = dest_dir / object_name
        
        shutil.copy2(file_path, dest_path)
        uri = f"file://{dest_path.absolute()}"
        logger.info(f"Uploaded to local: {uri}")
        return uri
    
    async def upload_fileobj(
        self,
        bucket_type: str,
        file_obj: BinaryIO,
        object_name: str,
        content_type: Optional[str] = None
    ) -> str:
        """
        Upload a file-like object to storage.
        
        Args:
            bucket_type: Type of storage bucket
            file_obj: File-like object to upload
            object_name: Name to store the file as
            content_type: MIME type of the file
            
        Returns:
            URI string
        """
        if self.is_s3_mode:
            bucket_name = self._get_bucket_name(bucket_type)
            extra_args = {}
            if content_type:
                extra_args['ContentType'] = content_type
            
            try:
                self._s3_client.upload_fileobj(
                    file_obj,
                    bucket_name,
                    object_name,
                    ExtraArgs=extra_args if extra_args else None
                )
                uri = f"s3://{bucket_name}/{object_name}"
                logger.info(f"Uploaded fileobj to S3: {uri}")
                return uri
            except ClientError as e:
                logger.error(f"S3 fileobj upload failed: {e}")
                raise
        else:
            dest_dir = self._ensure_local_dir(bucket_type)
            dest_path = dest_dir / object_name
            
            with open(dest_path, 'wb') as f:
                shutil.copyfileobj(file_obj, f)
            
            uri = f"file://{dest_path.absolute()}"
            logger.info(f"Uploaded fileobj to local: {uri}")
            return uri
    
    async def download_file(
        self,
        uri: str,
        dest_path: Union[str, Path]
    ) -> Path:
        """
        Download a file from storage.
        
        Args:
            uri: Storage URI (s3:// or file://)
            dest_path: Local path to download to
            
        Returns:
            Path to downloaded file
        """
        dest_path = Path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        if uri.startswith("s3://"):
            return await self._download_s3(uri, dest_path)
        elif uri.startswith("file://"):
            return await self._download_local(uri, dest_path)
        else:
            # Assume it's a local path
            src = Path(uri)
            if src.exists():
                shutil.copy2(src, dest_path)
                return dest_path
            raise ValueError(f"Invalid URI scheme: {uri}")
    
    async def _download_s3(self, uri: str, dest_path: Path) -> Path:
        """Download from S3/MinIO"""
        # Parse s3://bucket/key
        parts = uri.replace("s3://", "").split("/", 1)
        bucket_name = parts[0]
        object_key = parts[1] if len(parts) > 1 else ""
        
        try:
            self._s3_client.download_file(bucket_name, object_key, str(dest_path))
            logger.info(f"Downloaded from S3: {uri} -> {dest_path}")
            return dest_path
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    async def _download_local(self, uri: str, dest_path: Path) -> Path:
        """Copy from local storage"""
        src_path = Path(uri.replace("file://", ""))
        if not src_path.exists():
            raise FileNotFoundError(f"File not found: {src_path}")
        
        shutil.copy2(src_path, dest_path)
        logger.info(f"Downloaded from local: {uri} -> {dest_path}")
        return dest_path
    
    async def get_presigned_url(
        self,
        uri: str,
        expiry: timedelta = timedelta(hours=1)
    ) -> str:
        """
        Get a presigned URL for direct file access.
        
        Args:
            uri: Storage URI
            expiry: URL expiration time
            
        Returns:
            Presigned URL or local file path
        """
        if uri.startswith("s3://") and self.is_s3_mode:
            parts = uri.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else ""
            
            try:
                url = self._s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': bucket_name, 'Key': object_key},
                    ExpiresIn=int(expiry.total_seconds())
                )
                return url
            except ClientError as e:
                logger.error(f"Failed to generate presigned URL: {e}")
                raise
        else:
            # For local files, just return the path
            if uri.startswith("file://"):
                return uri.replace("file://", "")
            return uri
    
    async def delete_file(self, uri: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            uri: Storage URI
            
        Returns:
            True if deleted successfully
        """
        if uri.startswith("s3://") and self.is_s3_mode:
            parts = uri.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else ""
            
            try:
                self._s3_client.delete_object(Bucket=bucket_name, Key=object_key)
                logger.info(f"Deleted from S3: {uri}")
                return True
            except ClientError as e:
                logger.error(f"S3 delete failed: {e}")
                return False
        else:
            path = uri.replace("file://", "") if uri.startswith("file://") else uri
            try:
                Path(path).unlink()
                logger.info(f"Deleted local file: {uri}")
                return True
            except Exception as e:
                logger.error(f"Local delete failed: {e}")
                return False
    
    async def file_exists(self, uri: str) -> bool:
        """Check if a file exists in storage"""
        if uri.startswith("s3://") and self.is_s3_mode:
            parts = uri.replace("s3://", "").split("/", 1)
            bucket_name = parts[0]
            object_key = parts[1] if len(parts) > 1 else ""
            
            try:
                self._s3_client.head_object(Bucket=bucket_name, Key=object_key)
                return True
            except ClientError:
                return False
        else:
            path = uri.replace("file://", "") if uri.startswith("file://") else uri
            return Path(path).exists()
    
    async def list_files(
        self,
        bucket_type: str,
        prefix: str = ""
    ) -> list[dict]:
        """
        List files in a bucket/directory.
        
        Args:
            bucket_type: Type of storage bucket
            prefix: Filter by prefix/subdirectory
            
        Returns:
            List of file info dicts with name, size, modified
        """
        if self.is_s3_mode:
            bucket_name = self._get_bucket_name(bucket_type)
            
            try:
                response = self._s3_client.list_objects_v2(
                    Bucket=bucket_name,
                    Prefix=prefix
                )
                
                files = []
                for obj in response.get('Contents', []):
                    files.append({
                        'name': obj['Key'],
                        'size': obj['Size'],
                        'modified': obj['LastModified'].isoformat(),
                        'uri': f"s3://{bucket_name}/{obj['Key']}"
                    })
                return files
            except ClientError as e:
                logger.error(f"S3 list failed: {e}")
                return []
        else:
            dir_path = self.local_base / bucket_type
            if not dir_path.exists():
                return []
            
            files = []
            for p in dir_path.rglob("*"):
                if p.is_file() and (not prefix or p.name.startswith(prefix)):
                    stat = p.stat()
                    files.append({
                        'name': p.name,
                        'size': stat.st_size,
                        'modified': stat.st_mtime,
                        'uri': f"file://{p.absolute()}"
                    })
            return files
    
    def compute_checksum(self, file_path: Union[str, Path]) -> str:
        """Compute SHA256 checksum of a file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


# Global instance
storage_service = StorageService()
