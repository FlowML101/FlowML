"""
FlowML Data Format Handlers

Supports multiple data formats for personal use cases:

FINANCIAL:
- CSV (bank exports, credit card statements)
- OFX (Open Financial Exchange - bank downloads)
- QIF (Quicken Interchange Format)

HEALTH & FITNESS:
- XML (Apple Health exports)
- JSON (Fitbit, Google Fit exports)

PRODUCTIVITY:
- Excel (.xlsx, .xls - spreadsheets)
- Google Sheets (export as CSV/XLSX)
- ICS/iCal (calendar events)
- VCF/vCard (contacts)

DATA FORMATS:
- Parquet, Feather (columnar, fast)
- JSON, JSONL (API exports)
- YAML, TOML (config files)
- XML (generic)
- SQLite (local databases)
- LOG files (with pattern parsing)

Hardware-aware processing:
- Streaming for large files
- Chunk processing for memory efficiency
- Polars for speed (uses Rust under the hood)
"""
import io
import re
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import tempfile
import xml.etree.ElementTree as ET

import polars as pl
from loguru import logger


class DataFormat(str, Enum):
    """Supported data formats"""
    # Text/tabular
    CSV = "csv"
    TSV = "tsv"
    
    # Spreadsheets
    EXCEL = "excel"
    
    # Columnar/binary
    PARQUET = "parquet"
    FEATHER = "feather"
    
    # JSON variants
    JSON = "json"
    JSON_LINES = "jsonl"
    
    # Config formats
    YAML = "yaml"
    TOML = "toml"
    XML = "xml"
    
    # Financial
    OFX = "ofx"
    QIF = "qif"
    
    # Personal data
    ICAL = "ical"
    VCARD = "vcard"
    
    # Database
    SQLITE = "sqlite"
    
    # Logs
    LOG = "log"


# File extension to format mapping
EXTENSION_MAP = {
    # Text/CSV
    ".csv": DataFormat.CSV,
    ".tsv": DataFormat.TSV,
    ".txt": DataFormat.CSV,  # Assume CSV for .txt
    
    # Spreadsheets
    ".xlsx": DataFormat.EXCEL,
    ".xls": DataFormat.EXCEL,
    
    # Columnar
    ".parquet": DataFormat.PARQUET,
    ".pq": DataFormat.PARQUET,
    ".feather": DataFormat.FEATHER,
    ".arrow": DataFormat.FEATHER,
    ".ipc": DataFormat.FEATHER,
    
    # JSON
    ".json": DataFormat.JSON,
    ".jsonl": DataFormat.JSON_LINES,
    ".ndjson": DataFormat.JSON_LINES,
    
    # Config
    ".yaml": DataFormat.YAML,
    ".yml": DataFormat.YAML,
    ".toml": DataFormat.TOML,
    ".xml": DataFormat.XML,
    
    # Financial
    ".ofx": DataFormat.OFX,
    ".qfx": DataFormat.OFX,  # Quicken variant of OFX
    ".qif": DataFormat.QIF,
    
    # Personal
    ".ics": DataFormat.ICAL,
    ".ical": DataFormat.ICAL,
    ".vcf": DataFormat.VCARD,
    ".vcard": DataFormat.VCARD,
    
    # Database
    ".sqlite": DataFormat.SQLITE,
    ".db": DataFormat.SQLITE,
    ".sqlite3": DataFormat.SQLITE,
    
    # Logs
    ".log": DataFormat.LOG,
}


@dataclass
class ReadOptions:
    """Options for reading data files"""
    # CSV options
    delimiter: Optional[str] = None  # Auto-detect if None
    has_header: bool = True
    skip_rows: int = 0
    encoding: str = "utf-8"
    null_values: List[str] = None
    infer_schema_length: Optional[int] = None  # None = scan all rows for accurate type inference
    ignore_errors: bool = True  # Ignore parsing errors for mixed types and edge cases
    
    # Sampling
    sample_rows: Optional[int] = None  # Read only N rows
    
    # Memory management
    low_memory: bool = False  # Stream processing for large files
    chunk_size: int = 100_000  # Rows per chunk in low_memory mode
    
    # Schema
    schema: Optional[Dict[str, Any]] = None  # Column type hints
    
    # XML/Health data options
    xml_record_path: Optional[str] = None  # XPath to records (e.g., "Record" for Apple Health)
    
    # Log parsing
    log_pattern: Optional[str] = None  # Regex pattern for log parsing
    
    def __post_init__(self):
        if self.null_values is None:
            self.null_values = ["", "NA", "N/A", "null", "NULL", "None", "nan", "NaN", "."]


@dataclass
class WriteOptions:
    """Options for writing data files"""
    # CSV options
    delimiter: str = ","
    include_header: bool = True
    
    # Compression
    compression: Optional[str] = None  # "gzip", "snappy", "zstd", "lz4"
    
    # Parquet options
    row_group_size: int = 100_000


class DataReader:
    """
    Universal data file reader with format auto-detection.
    
    Uses Polars for fast, memory-efficient reading.
    Supports streaming for files larger than available RAM.
    """
    
    @staticmethod
    def detect_format(file_path: Union[str, Path]) -> DataFormat:
        """Detect file format from extension"""
        ext = Path(file_path).suffix.lower()
        if ext in EXTENSION_MAP:
            return EXTENSION_MAP[ext]
        raise ValueError(f"Unsupported file format: {ext}")
    
    @staticmethod
    def detect_csv_delimiter(file_path: Union[str, Path], sample_bytes: int = 8192) -> str:
        """Auto-detect CSV delimiter by sampling file"""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(sample_bytes)
        
        # Count potential delimiters
        delimiters = {
            ",": sample.count(","),
            ";": sample.count(";"),
            "\t": sample.count("\t"),
            "|": sample.count("|"),
        }
        
        # Return most common (excluding newlines)
        return max(delimiters, key=delimiters.get)
    
    @classmethod
    def read(
        cls,
        file_path: Union[str, Path],
        options: Optional[ReadOptions] = None
    ) -> pl.DataFrame:
        """
        Read a data file into a Polars DataFrame.
        
        Auto-detects format from extension and applies appropriate reader.
        """
        options = options or ReadOptions()
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        format_type = cls.detect_format(file_path)
        logger.info(f"Reading {format_type.value} file: {file_path.name}")
        
        readers = {
            DataFormat.CSV: cls._read_csv,
            DataFormat.TSV: cls._read_tsv,
            DataFormat.EXCEL: cls._read_excel,
            DataFormat.PARQUET: cls._read_parquet,
            DataFormat.JSON: cls._read_json,
            DataFormat.JSON_LINES: cls._read_jsonl,
            DataFormat.FEATHER: cls._read_feather,
            DataFormat.SQLITE: cls._read_sqlite,
            DataFormat.XML: cls._read_xml,
            DataFormat.YAML: cls._read_yaml,
            DataFormat.TOML: cls._read_toml,
            DataFormat.OFX: cls._read_ofx,
            DataFormat.QIF: cls._read_qif,
            DataFormat.ICAL: cls._read_ical,
            DataFormat.VCARD: cls._read_vcard,
            DataFormat.LOG: cls._read_log,
        }
        
        reader = readers.get(format_type)
        if not reader:
            raise ValueError(f"No reader for format: {format_type}")
        
        df = reader(file_path, options)
        
        # Apply sampling if requested
        if options.sample_rows and len(df) > options.sample_rows:
            df = df.head(options.sample_rows)
        
        logger.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
        return df
    
    @classmethod
    def _read_csv(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read CSV file"""
        delimiter = options.delimiter or cls.detect_csv_delimiter(file_path)
        
        return pl.read_csv(
            file_path,
            separator=delimiter,
            has_header=options.has_header,
            skip_rows=options.skip_rows,
            encoding=options.encoding,
            null_values=options.null_values,
            low_memory=options.low_memory,
            n_rows=options.sample_rows,
            infer_schema_length=options.infer_schema_length,
            ignore_errors=options.ignore_errors,
        )
    
    @classmethod
    def _read_tsv(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read TSV file"""
        return pl.read_csv(
            file_path,
            separator="\t",
            has_header=options.has_header,
            skip_rows=options.skip_rows,
            encoding=options.encoding,
            null_values=options.null_values,
            n_rows=options.sample_rows,
            infer_schema_length=options.infer_schema_length,
            ignore_errors=options.ignore_errors,
        )
    
    @classmethod
    def _read_excel(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read Excel file (.xlsx, .xls)"""
        try:
            # Polars can read Excel via calamine
            return pl.read_excel(
                file_path,
                sheet_id=1,  # First sheet
                read_options={"skip_rows": options.skip_rows, "has_header": options.has_header}
            )
        except ImportError:
            # Fallback to pandas if calamine not available
            logger.warning("calamine not installed, falling back to pandas for Excel")
            import pandas as pd
            pdf = pd.read_excel(
                file_path,
                header=0 if options.has_header else None,
                skiprows=options.skip_rows,
            )
            return pl.from_pandas(pdf)
    
    @classmethod
    def _read_parquet(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read Parquet file"""
        df = pl.read_parquet(file_path)
        if options.sample_rows:
            df = df.head(options.sample_rows)
        return df
    
    @classmethod
    def _read_json(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read JSON file (array of records)"""
        with open(file_path, "r", encoding=options.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pl.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find array in dict
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    return pl.DataFrame(value)
            # Single record
            return pl.DataFrame([data])
        else:
            raise ValueError("JSON must be an array or object")
    
    @classmethod
    def _read_jsonl(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read JSON Lines file (one JSON object per line)"""
        return pl.read_ndjson(file_path)
    
    @classmethod
    def _read_feather(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read Feather/Arrow IPC file"""
        return pl.read_ipc(file_path)
    
    @classmethod
    def _read_sqlite(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read from SQLite database (first table)"""
        import sqlite3
        
        conn = sqlite3.connect(file_path)
        cursor = conn.cursor()
        
        # Get first table name
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name LIMIT 1")
        table = cursor.fetchone()
        if not table:
            raise ValueError("No tables found in SQLite database")
        table_name = table[0]
        
        # Read table
        query = f"SELECT * FROM {table_name}"
        if options.sample_rows:
            query += f" LIMIT {options.sample_rows}"
        
        import pandas as pd
        pdf = pd.read_sql_query(query, conn)
        conn.close()
        
        return pl.from_pandas(pdf)
    
    @classmethod
    def _read_xml(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read XML file - supports generic XML and Apple Health exports.
        
        For Apple Health: Looks for 'Record' elements with health data.
        For generic XML: Flattens first-level child elements to rows.
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Check if it's Apple Health export
        if root.tag == "HealthData" or "Record" in [child.tag for child in root]:
            return cls._parse_apple_health_xml(root, options)
        
        # Generic XML: use specified record path or auto-detect
        record_tag = options.xml_record_path
        if not record_tag:
            # Find most common child tag
            child_tags = [child.tag for child in root]
            if child_tags:
                record_tag = max(set(child_tags), key=child_tags.count)
        
        if not record_tag:
            raise ValueError("Could not detect record elements in XML")
        
        # Extract records
        records = []
        for elem in root.iter(record_tag):
            record = dict(elem.attrib)  # Get attributes
            # Also get child text elements
            for child in elem:
                if child.text and child.text.strip():
                    record[child.tag] = child.text.strip()
            if record:
                records.append(record)
        
        if not records:
            raise ValueError(f"No records found with tag: {record_tag}")
        
        return pl.DataFrame(records)
    
    @classmethod
    def _parse_apple_health_xml(cls, root: ET.Element, options: ReadOptions) -> pl.DataFrame:
        """Parse Apple Health export XML"""
        records = []
        
        for record in root.iter("Record"):
            data = dict(record.attrib)
            # Parse dates
            if "startDate" in data:
                data["startDate"] = data["startDate"][:19]  # Trim timezone
            if "endDate" in data:
                data["endDate"] = data["endDate"][:19]
            if "creationDate" in data:
                data["creationDate"] = data["creationDate"][:19]
            records.append(data)
            
            if options.sample_rows and len(records) >= options.sample_rows:
                break
        
        if not records:
            # Try Workout records
            for workout in root.iter("Workout"):
                records.append(dict(workout.attrib))
                if options.sample_rows and len(records) >= options.sample_rows:
                    break
        
        if not records:
            raise ValueError("No health records found in XML")
        
        return pl.DataFrame(records)
    
    @classmethod
    def _read_yaml(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read YAML file"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        
        with open(file_path, "r", encoding=options.encoding) as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            return pl.DataFrame(data)
        elif isinstance(data, dict):
            # Try to find list in dict
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    return pl.DataFrame(value)
            # Single record or key-value pairs
            return pl.DataFrame([data])
        else:
            raise ValueError("YAML must contain a list or object")
    
    @classmethod
    def _read_toml(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """Read TOML file"""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("tomli required for TOML files: pip install tomli")
        
        with open(file_path, "rb") as f:
            data = tomllib.load(f)
        
        # TOML is usually nested, try to find tabular data
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                return pl.DataFrame(value)
        
        # Flatten to single row
        return pl.DataFrame([data])
    
    @classmethod
    def _read_ofx(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read OFX (Open Financial Exchange) bank statement file.
        
        OFX is used by most banks for downloadable statements.
        Extracts transaction records (STMTTRN elements).
        """
        with open(file_path, "r", encoding=options.encoding, errors="ignore") as f:
            content = f.read()
        
        transactions = []
        
        # OFX can be SGML-like or XML - handle both
        # Find all STMTTRN (statement transaction) blocks
        pattern = r"<STMTTRN>(.*?)</STMTTRN>"
        matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            # Try SGML format without closing tags
            pattern = r"<STMTTRN>(.*?)(?=<STMTTRN>|</BANKTRANLIST>|$)"
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            txn = {}
            # Extract common fields
            fields = ["TRNTYPE", "DTPOSTED", "TRNAMT", "FITID", "NAME", "MEMO", "CHECKNUM"]
            for field in fields:
                # Try XML style
                field_match = re.search(f"<{field}>(.*?)</{field}>", match, re.IGNORECASE)
                if not field_match:
                    # Try SGML style (no closing tag)
                    field_match = re.search(f"<{field}>([^<\n]+)", match, re.IGNORECASE)
                if field_match:
                    txn[field.lower()] = field_match.group(1).strip()
            
            if txn:
                # Parse amount as float
                if "trnamt" in txn:
                    try:
                        txn["amount"] = float(txn["trnamt"])
                        del txn["trnamt"]
                    except:
                        pass
                
                # Parse date (YYYYMMDD format)
                if "dtposted" in txn:
                    try:
                        date_str = txn["dtposted"][:8]
                        txn["date"] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        del txn["dtposted"]
                    except:
                        pass
                
                transactions.append(txn)
        
        if not transactions:
            raise ValueError("No transactions found in OFX file")
        
        return pl.DataFrame(transactions)
    
    @classmethod
    def _read_qif(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read QIF (Quicken Interchange Format) file.
        
        QIF is an older format still used by some financial software.
        """
        with open(file_path, "r", encoding=options.encoding, errors="ignore") as f:
            content = f.read()
        
        transactions = []
        current_txn = {}
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("^"):
                # End of record
                if current_txn:
                    transactions.append(current_txn)
                    current_txn = {}
            elif line.startswith("D"):
                # Date
                current_txn["date"] = line[1:]
            elif line.startswith("T") or line.startswith("U"):
                # Amount
                try:
                    amount_str = line[1:].replace(",", "").replace("$", "")
                    current_txn["amount"] = float(amount_str)
                except:
                    current_txn["amount_raw"] = line[1:]
            elif line.startswith("P"):
                # Payee
                current_txn["payee"] = line[1:]
            elif line.startswith("M"):
                # Memo
                current_txn["memo"] = line[1:]
            elif line.startswith("C"):
                # Cleared status
                current_txn["cleared"] = line[1:]
            elif line.startswith("N"):
                # Check number or category
                current_txn["number"] = line[1:]
            elif line.startswith("L"):
                # Category
                current_txn["category"] = line[1:]
        
        # Don't forget last transaction
        if current_txn:
            transactions.append(current_txn)
        
        if not transactions:
            raise ValueError("No transactions found in QIF file")
        
        return pl.DataFrame(transactions)
    
    @classmethod
    def _read_ical(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read iCalendar (.ics) file - calendar events.
        
        Extracts VEVENT components into tabular format.
        """
        with open(file_path, "r", encoding=options.encoding, errors="ignore") as f:
            content = f.read()
        
        events = []
        current_event = {}
        in_event = False
        
        for line in content.split("\n"):
            line = line.strip()
            
            if line == "BEGIN:VEVENT":
                in_event = True
                current_event = {}
            elif line == "END:VEVENT":
                in_event = False
                if current_event:
                    events.append(current_event)
                    current_event = {}
            elif in_event and ":" in line:
                # Handle property;params:value format
                prop_part, _, value = line.partition(":")
                prop = prop_part.split(";")[0]  # Remove parameters
                
                # Map common properties
                prop_map = {
                    "SUMMARY": "title",
                    "DTSTART": "start",
                    "DTEND": "end",
                    "DESCRIPTION": "description",
                    "LOCATION": "location",
                    "ORGANIZER": "organizer",
                    "STATUS": "status",
                    "UID": "uid",
                }
                
                key = prop_map.get(prop, prop.lower())
                
                # Parse dates
                if key in ("start", "end") and value:
                    # Try to format nicely
                    if len(value) >= 8:
                        date_part = value[:8]
                        formatted = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                        if len(value) > 8 and "T" in value:
                            time_part = value.split("T")[1][:6]
                            if len(time_part) >= 4:
                                formatted += f" {time_part[:2]}:{time_part[2:4]}"
                        value = formatted
                
                current_event[key] = value
        
        if not events:
            raise ValueError("No events found in iCal file")
        
        return pl.DataFrame(events)
    
    @classmethod
    def _read_vcard(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read vCard (.vcf) file - contacts.
        
        Extracts contact information into tabular format.
        """
        with open(file_path, "r", encoding=options.encoding, errors="ignore") as f:
            content = f.read()
        
        contacts = []
        current_contact = {}
        in_contact = False
        
        for line in content.split("\n"):
            line = line.strip()
            
            if line == "BEGIN:VCARD":
                in_contact = True
                current_contact = {}
            elif line == "END:VCARD":
                in_contact = False
                if current_contact:
                    contacts.append(current_contact)
                    current_contact = {}
            elif in_contact and ":" in line:
                prop_part, _, value = line.partition(":")
                prop = prop_part.split(";")[0]  # Remove parameters
                
                # Map common properties
                if prop == "FN":
                    current_contact["full_name"] = value
                elif prop == "N":
                    parts = value.split(";")
                    if len(parts) >= 2:
                        current_contact["last_name"] = parts[0]
                        current_contact["first_name"] = parts[1]
                elif prop == "EMAIL":
                    # Handle multiple emails
                    if "email" in current_contact:
                        current_contact["email2"] = value
                    else:
                        current_contact["email"] = value
                elif prop == "TEL":
                    # Handle multiple phones
                    if "phone" in current_contact:
                        current_contact["phone2"] = value
                    else:
                        current_contact["phone"] = value
                elif prop == "ORG":
                    current_contact["organization"] = value
                elif prop == "TITLE":
                    current_contact["title"] = value
                elif prop == "ADR":
                    # Address: PO;Ext;Street;City;Region;Postal;Country
                    parts = value.split(";")
                    if len(parts) >= 7:
                        current_contact["city"] = parts[3]
                        current_contact["state"] = parts[4]
                        current_contact["postal_code"] = parts[5]
                        current_contact["country"] = parts[6]
                elif prop == "BDAY":
                    current_contact["birthday"] = value
                elif prop == "NOTE":
                    current_contact["notes"] = value
        
        if not contacts:
            raise ValueError("No contacts found in vCard file")
        
        return pl.DataFrame(contacts)
    
    @classmethod
    def _read_log(cls, file_path: Path, options: ReadOptions) -> pl.DataFrame:
        """
        Read log file with pattern detection.
        
        Attempts to parse common log formats:
        - Apache/Nginx access logs
        - Syslog format
        - Generic timestamped logs
        """
        with open(file_path, "r", encoding=options.encoding, errors="ignore") as f:
            lines = f.readlines()
        
        if options.log_pattern:
            # Use custom pattern
            pattern = re.compile(options.log_pattern)
            records = []
            for line in lines:
                match = pattern.match(line)
                if match:
                    records.append(match.groupdict())
            if records:
                return pl.DataFrame(records)
        
        # Try common patterns
        patterns = [
            # Apache/Nginx combined log format
            (
                r'(?P<ip>[\d.]+)\s+-\s+(?P<user>\S+)\s+\[(?P<timestamp>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<path>\S+)\s+(?P<protocol>[^"]+)"\s+(?P<status>\d+)\s+(?P<size>\d+)',
                "apache"
            ),
            # Syslog format
            (
                r'(?P<timestamp>\w+\s+\d+\s+[\d:]+)\s+(?P<host>\S+)\s+(?P<process>\S+):\s+(?P<message>.+)',
                "syslog"
            ),
            # Generic timestamp + message
            (
                r'(?P<timestamp>[\d\-T:\.]+)\s*[\|\-]?\s*(?P<level>\w+)?\s*[\|\-]?\s*(?P<message>.+)',
                "generic"
            ),
        ]
        
        for pattern, name in patterns:
            regex = re.compile(pattern)
            records = []
            for line in lines[:100]:  # Sample first 100 lines
                match = regex.match(line.strip())
                if match:
                    records.append(match.groupdict())
            
            if len(records) > len(lines[:100]) * 0.5:  # >50% match rate
                # Parse full file with this pattern
                records = []
                for line in lines:
                    match = regex.match(line.strip())
                    if match:
                        records.append(match.groupdict())
                        if options.sample_rows and len(records) >= options.sample_rows:
                            break
                
                logger.info(f"Detected log format: {name}")
                return pl.DataFrame(records)
        
        # Fallback: just split by common delimiters or return as lines
        if lines:
            return pl.DataFrame({"line_number": range(1, len(lines) + 1), "content": [l.strip() for l in lines]})
        
        raise ValueError("Could not parse log file")
    
    @classmethod
    def read_bytes(
        cls,
        content: bytes,
        filename: str,
        options: Optional[ReadOptions] = None
    ) -> pl.DataFrame:
        """
        Read from bytes (for uploaded files).
        
        Creates temp file for formats that need file access.
        """
        options = options or ReadOptions()
        format_type = cls.detect_format(filename)
        
        # Some formats can read directly from bytes
        if format_type == DataFormat.CSV:
            delimiter = options.delimiter or ","
            return pl.read_csv(
                io.BytesIO(content),
                separator=delimiter,
                has_header=options.has_header,
                null_values=options.null_values,
            )
        elif format_type == DataFormat.JSON:
            data = json.loads(content.decode(options.encoding))
            return pl.DataFrame(data if isinstance(data, list) else [data])
        elif format_type == DataFormat.JSON_LINES:
            return pl.read_ndjson(io.BytesIO(content))
        elif format_type == DataFormat.PARQUET:
            return pl.read_parquet(io.BytesIO(content))
        else:
            # Write to temp file for other formats
            ext = Path(filename).suffix
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(content)
                temp_path = f.name
            
            try:
                return cls.read(temp_path, options)
            finally:
                Path(temp_path).unlink()


class DataWriter:
    """
    Universal data file writer.
    
    Supports multiple output formats with compression options.
    """
    
    @classmethod
    def write(
        cls,
        df: pl.DataFrame,
        file_path: Union[str, Path],
        format_type: Optional[DataFormat] = None,
        options: Optional[WriteOptions] = None
    ) -> Path:
        """
        Write DataFrame to file.
        
        Auto-detects format from extension if not specified.
        """
        options = options or WriteOptions()
        file_path = Path(file_path)
        
        if format_type is None:
            format_type = DataReader.detect_format(file_path)
        
        logger.info(f"Writing {format_type.value} file: {file_path.name}")
        
        writers = {
            DataFormat.CSV: cls._write_csv,
            DataFormat.TSV: cls._write_tsv,
            DataFormat.PARQUET: cls._write_parquet,
            DataFormat.JSON: cls._write_json,
            DataFormat.JSON_LINES: cls._write_jsonl,
            DataFormat.FEATHER: cls._write_feather,
        }
        
        writer = writers.get(format_type)
        if not writer:
            raise ValueError(f"No writer for format: {format_type}")
        
        writer(df, file_path, options)
        logger.info(f"Wrote {len(df):,} rows to {file_path}")
        return file_path
    
    @classmethod
    def _write_csv(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_csv(
            file_path,
            separator=options.delimiter,
            include_header=options.include_header,
        )
    
    @classmethod
    def _write_tsv(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_csv(
            file_path,
            separator="\t",
            include_header=options.include_header,
        )
    
    @classmethod
    def _write_parquet(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_parquet(
            file_path,
            compression=options.compression or "snappy",
            row_group_size=options.row_group_size,
        )
    
    @classmethod
    def _write_json(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_json(file_path, row_oriented=True)
    
    @classmethod
    def _write_jsonl(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_ndjson(file_path)
    
    @classmethod
    def _write_feather(cls, df: pl.DataFrame, file_path: Path, options: WriteOptions):
        df.write_ipc(file_path, compression=options.compression or "zstd")
    
    @classmethod
    def to_bytes(
        cls,
        df: pl.DataFrame,
        format_type: DataFormat,
        options: Optional[WriteOptions] = None
    ) -> bytes:
        """
        Write DataFrame to bytes for download.
        """
        options = options or WriteOptions()
        buffer = io.BytesIO()
        
        if format_type == DataFormat.CSV:
            df.write_csv(buffer, separator=options.delimiter, include_header=options.include_header)
        elif format_type == DataFormat.PARQUET:
            df.write_parquet(buffer, compression=options.compression or "snappy")
        elif format_type == DataFormat.JSON:
            buffer.write(df.write_json(row_oriented=True).encode())
        elif format_type == DataFormat.JSON_LINES:
            df.write_ndjson(buffer)
        else:
            raise ValueError(f"Cannot convert to bytes: {format_type}")
        
        buffer.seek(0)
        return buffer.read()


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a data file without fully loading it.
    
    Returns metadata like row count, columns, types, etc.
    """
    file_path = Path(file_path)
    format_type = DataReader.detect_format(file_path)
    
    info = {
        "path": str(file_path),
        "name": file_path.name,
        "format": format_type.value,
        "size_bytes": file_path.stat().st_size,
        "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
    }
    
    # Quick schema scan
    try:
        if format_type == DataFormat.PARQUET:
            # Parquet has metadata we can read without loading data
            schema = pl.read_parquet_schema(file_path)
            info["columns"] = list(schema.keys())
            info["dtypes"] = {k: str(v) for k, v in schema.items()}
            info["num_columns"] = len(schema)
        else:
            # For other formats, read first few rows
            df = DataReader.read(file_path, ReadOptions(sample_rows=5))
            info["columns"] = df.columns
            info["dtypes"] = {col: str(df[col].dtype) for col in df.columns}
            info["num_columns"] = len(df.columns)
            
    except Exception as e:
        logger.warning(f"Could not scan file schema: {e}")
    
    return info


# Convenience functions
def read_data(path: Union[str, Path], **kwargs) -> pl.DataFrame:
    """Quick read with default options"""
    options = ReadOptions(**kwargs) if kwargs else None
    return DataReader.read(path, options)


def write_data(df: pl.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    """Quick write with default options"""
    options = WriteOptions(**kwargs) if kwargs else None
    return DataWriter.write(df, path, options=options)
