"""
Logs Router - Log retrieval and streaming for the Logs page
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import re
from loguru import logger

from config import settings

router = APIRouter(prefix="/logs", tags=["logs"])


class LogEntry(BaseModel):
    """Single log entry"""
    timestamp: datetime
    level: str
    source: str
    message: str
    line: Optional[int] = None


class LogsResponse(BaseModel):
    """Logs response with pagination"""
    logs: List[LogEntry]
    total: int
    page: int
    page_size: int
    has_more: bool


class LogStats(BaseModel):
    """Log statistics"""
    total_entries: int
    by_level: dict
    oldest: Optional[datetime]
    newest: Optional[datetime]
    log_files: List[str]


# Regex to parse loguru log format
LOG_PATTERN = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\|\s*(\w+)\s*\|\s*([^|]+)\s*-\s*(.*)"
)


def parse_log_line(line: str) -> Optional[LogEntry]:
    """Parse a single log line"""
    match = LOG_PATTERN.match(line.strip())
    if match:
        timestamp_str, level, source, message = match.groups()
        try:
            # Parse timestamp (loguru format: 2026-02-01 12:22:37.465)
            timestamp = datetime.strptime(timestamp_str[:19], "%Y-%m-%d %H:%M:%S")
            return LogEntry(
                timestamp=timestamp,
                level=level.strip(),
                source=source.strip(),
                message=message.strip()
            )
        except ValueError:
            pass
    return None


def get_log_files() -> List[Path]:
    """Get all log files sorted by modification time"""
    log_dir = settings.BASE_DIR / "logs"
    if not log_dir.exists():
        return []
    
    files = list(log_dir.glob("flowml_*.log"))
    return sorted(files, key=lambda f: f.stat().st_mtime, reverse=True)


@router.get("", response_model=LogsResponse)
async def get_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=10, le=500),
    level: Optional[str] = Query(None, description="Filter by level: DEBUG, INFO, WARNING, ERROR"),
    source: Optional[str] = Query(None, description="Filter by source module"),
    search: Optional[str] = Query(None, description="Search in message"),
    since: Optional[datetime] = Query(None, description="Logs since timestamp"),
    until: Optional[datetime] = Query(None, description="Logs until timestamp"),
):
    """
    Get paginated logs with filtering.
    """
    log_files = get_log_files()
    if not log_files:
        return LogsResponse(logs=[], total=0, page=page, page_size=page_size, has_more=False)
    
    all_entries = []
    
    # Read from most recent log file first
    for log_file in log_files[:3]:  # Limit to 3 most recent files
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    entry = parse_log_line(line)
                    if entry:
                        entry.line = line_num
                        
                        # Apply filters
                        if level and entry.level.upper() != level.upper():
                            continue
                        if source and source.lower() not in entry.source.lower():
                            continue
                        if search and search.lower() not in entry.message.lower():
                            continue
                        if since and entry.timestamp < since:
                            continue
                        if until and entry.timestamp > until:
                            continue
                        
                        all_entries.append(entry)
        except Exception as e:
            logger.warning(f"Failed to read log file {log_file}: {e}")
    
    # Sort by timestamp descending (newest first)
    all_entries.sort(key=lambda e: e.timestamp, reverse=True)
    
    # Paginate
    total = len(all_entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = all_entries[start:end]
    
    return LogsResponse(
        logs=page_entries,
        total=total,
        page=page,
        page_size=page_size,
        has_more=end < total
    )


@router.get("/stats", response_model=LogStats)
async def get_log_stats():
    """Get log statistics"""
    log_files = get_log_files()
    
    by_level = {"DEBUG": 0, "INFO": 0, "WARNING": 0, "ERROR": 0}
    oldest = None
    newest = None
    total = 0
    
    for log_file in log_files[:3]:
        try:
            with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    entry = parse_log_line(line)
                    if entry:
                        total += 1
                        level = entry.level.upper()
                        if level in by_level:
                            by_level[level] += 1
                        
                        if oldest is None or entry.timestamp < oldest:
                            oldest = entry.timestamp
                        if newest is None or entry.timestamp > newest:
                            newest = entry.timestamp
        except Exception:
            pass
    
    return LogStats(
        total_entries=total,
        by_level=by_level,
        oldest=oldest,
        newest=newest,
        log_files=[f.name for f in log_files]
    )


@router.get("/recent", response_model=List[LogEntry])
async def get_recent_logs(
    count: int = Query(50, ge=1, le=200),
    level: Optional[str] = None,
):
    """Get most recent N log entries (for live monitoring)"""
    log_files = get_log_files()
    if not log_files:
        return []
    
    entries = []
    
    # Read from most recent log file
    try:
        with open(log_files[0], "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        # Process from end
        for line in reversed(lines):
            entry = parse_log_line(line)
            if entry:
                if level and entry.level.upper() != level.upper():
                    continue
                entries.append(entry)
                if len(entries) >= count:
                    break
    except Exception as e:
        logger.warning(f"Failed to read recent logs: {e}")
    
    return entries


@router.delete("/clear")
async def clear_old_logs(
    keep_days: int = Query(7, ge=1, le=30, description="Keep logs from last N days")
):
    """Delete log files older than specified days"""
    log_files = get_log_files()
    cutoff = datetime.utcnow() - timedelta(days=keep_days)
    deleted = []
    
    for log_file in log_files:
        try:
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            if mtime < cutoff:
                log_file.unlink()
                deleted.append(log_file.name)
        except Exception as e:
            logger.warning(f"Failed to delete {log_file}: {e}")
    
    return {
        "deleted": deleted,
        "count": len(deleted),
        "kept": len(log_files) - len(deleted)
    }
