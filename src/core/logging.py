"""Enhanced logging configuration for production use"""

import logging
import logging.handlers
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager
import threading
import uuid


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
        }
        
        # Add extra fields
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
            
        if hasattr(record, 'provider'):
            log_data['provider'] = record.provider
            
        if hasattr(record, 'resource_type'):
            log_data['resource_type'] = record.resource_type
            
        if hasattr(record, 'region'):
            log_data['region'] = record.region
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class SecurityFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'secret', 'token', 'key', 'api_key',
        'access_key', 'private_key', 'credential', 'auth'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Redact sensitive information from log records"""
        message = record.getMessage().lower()
        
        # Check for sensitive patterns
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Redact the value after the sensitive key
                record.msg = self._redact_message(record.msg, pattern)
        
        return True
    
    def _redact_message(self, message: str, pattern: str) -> str:
        """Redact sensitive values in message"""
        import re
        # Patterns to match key=value, key:value, etc.
        patterns = [
            rf'{pattern}["\']?\s*[:=]\s*["\']?([^"\'\s,}}]+)',
            rf'"?{pattern}"?\s*:\s*"([^"]+)"',
        ]
        
        for p in patterns:
            message = re.sub(p, f'{pattern}=***REDACTED***', message, flags=re.IGNORECASE)
        
        return message


class AuditLogger:
    """Specialized logger for audit events"""
    
    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(logging.INFO)
        
        if log_file:
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=10
            )
            handler.setFormatter(StructuredFormatter())
            self.logger.addHandler(handler)
    
    def log_event(self, event_type: str, user: str, action: str, 
                  resource: Optional[str] = None, result: str = "success",
                  details: Optional[Dict[str, Any]] = None):
        """Log an audit event"""
        extra = {
            'event_type': event_type,
            'user': user,
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        if result == "success":
            self.logger.info(f"Audit: {event_type} - {action}", extra=extra)
        else:
            self.logger.warning(f"Audit: {event_type} - {action} failed", extra=extra)


class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        self.logger.setLevel(logging.INFO)
        self._timers = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """Context manager to time operations"""
        start_time = datetime.utcnow()
        timer_id = str(uuid.uuid4())
        
        with self._lock:
            self._timers[timer_id] = {
                'operation': operation,
                'start_time': start_time,
                **kwargs
            }
        
        try:
            yield timer_id
        finally:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            with self._lock:
                timer_info = self._timers.pop(timer_id, {})
            
            self.logger.info(
                f"Performance: {operation} completed in {duration:.3f}s",
                extra={
                    'operation': operation,
                    'duration': duration,
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    **kwargs
                }
            )
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", **tags):
        """Log a custom metric"""
        self.logger.info(
            f"Metric: {metric_name}={value}{unit}",
            extra={
                'metric_name': metric_name,
                'value': value,
                'unit': unit,
                'tags': tags,
                'timestamp': datetime.utcnow().isoformat()
            }
        )


class LoggerManager:
    """Centralized logger management"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loggers = {}
        self.audit_logger = None
        self.performance_logger = None
        self._request_context = threading.local()
    
    def setup_logging(self, 
                     level: str = "INFO",
                     log_file: Optional[Path] = None,
                     structured: bool = False,
                     console: bool = True,
                     audit_file: Optional[Path] = None):
        """Setup application-wide logging configuration"""
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        root_logger.handlers = []
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            if structured:
                console_handler.setFormatter(StructuredFormatter())
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            console_handler.addFilter(SecurityFilter())
            root_logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10485760,  # 10MB
                backupCount=5
            )
            if structured:
                file_handler.setFormatter(StructuredFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                    )
                )
            file_handler.addFilter(SecurityFilter())
            root_logger.addHandler(file_handler)
        
        # Setup specialized loggers
        self.audit_logger = AuditLogger(audit_file)
        self.performance_logger = PerformanceLogger()
        
        # Configure third-party loggers
        logging.getLogger('boto3').setLevel(logging.WARNING)
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('azure').setLevel(logging.WARNING)
        logging.getLogger('google').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a logger"""
        if name not in self.loggers:
            logger = logging.getLogger(name)
            self.loggers[name] = logger
        return self.loggers[name]
    
    def set_request_context(self, request_id: str, **kwargs):
        """Set request context for current thread"""
        self._request_context.request_id = request_id
        for key, value in kwargs.items():
            setattr(self._request_context, key, value)
    
    def clear_request_context(self):
        """Clear request context"""
        for key in list(self._request_context.__dict__.keys()):
            delattr(self._request_context, key)
    
    @contextmanager
    def request_context(self, request_id: Optional[str] = None, **kwargs):
        """Context manager for request-scoped logging"""
        request_id = request_id or str(uuid.uuid4())
        self.set_request_context(request_id, **kwargs)
        try:
            yield request_id
        finally:
            self.clear_request_context()
    
    def log_with_context(self, logger_name: str, level: str, message: str, **extra):
        """Log with current request context"""
        logger = self.get_logger(logger_name)
        
        # Add request context
        if hasattr(self._request_context, 'request_id'):
            extra['request_id'] = self._request_context.request_id
        
        for key in self._request_context.__dict__:
            if key != 'request_id':
                extra[key] = getattr(self._request_context, key)
        
        getattr(logger, level.lower())(message, extra=extra)


# Global logger manager instance
logger_manager = LoggerManager()


def setup_logging(**kwargs):
    """Setup logging for the application"""
    logger_manager.setup_logging(**kwargs)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logger_manager.get_logger(name)


def get_audit_logger() -> AuditLogger:
    """Get audit logger instance"""
    return logger_manager.audit_logger


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance"""
    return logger_manager.performance_logger