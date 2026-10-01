# src/config/logging.py
"""Structured logging configuration."""

import logging
import json
import sys
from datetime import datetime
from pythonjsonlogger import jsonlogger


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)


def setup_logging(log_level: str = "INFO") -> None:
    """Setup structured JSON logging."""
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # JSON handler (stdout)
    json_handler = logging.StreamHandler(sys.stdout)
    json_handler.setFormatter(CustomJsonFormatter())
    root_logger.addHandler(json_handler)
    
    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)