"""Security utilities and middleware"""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, Any, List, Callable
from datetime import datetime, timedelta
from functools import wraps
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
import base64
import json
import logging

from .exceptions import AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
        self._cleanup_interval = 300  # Cleanup every 5 minutes
        self._last_cleanup = time.time()
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()
        
        # Cleanup old entries periodically
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup()
        
        # Initialize or get request history
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests outside the window
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True
    
    def get_reset_time(self, identifier: str) -> int:
        """Get time until rate limit resets"""
        if identifier not in self.requests or not self.requests[identifier]:
            return 0
        
        oldest_request = min(self.requests[identifier])
        reset_time = oldest_request + self.window_seconds
        current_time = time.time()
        
        return max(0, int(reset_time - current_time))
    
    def _cleanup(self):
        """Remove old entries"""
        current_time = time.time()
        for identifier in list(self.requests.keys()):
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier]
                if current_time - req_time < self.window_seconds
            ]
            if not self.requests[identifier]:
                del self.requests[identifier]
        self._last_cleanup = current_time


class TokenManager:
    """JWT token management"""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = 'HS256'):
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token"""
        payload = payload.copy()
        payload['exp'] = datetime.utcnow() + timedelta(seconds=expires_in)
        payload['iat'] = datetime.utcnow()
        payload['jti'] = secrets.token_urlsafe(16)  # JWT ID for tracking
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def revoke_token(self, token_id: str):
        """Add token to revocation list (implement with cache/db)"""
        # This would typically be stored in Redis or database
        pass


class EncryptionManager:
    """Handle encryption/decryption of sensitive data"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.fernet = Fernet(master_key.encode() if isinstance(master_key, str) else master_key)
        else:
            self.fernet = Fernet(Fernet.generate_key())
    
    @classmethod
    def generate_key(cls, password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password"""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt(self, data: str) -> str:
        """Encrypt string data"""
        if not isinstance(data, bytes):
            data = data.encode()
        encrypted = self.fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt string data"""
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Failed to decrypt data")
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt dictionary data"""
        json_str = json.dumps(data)
        return self.encrypt(json_str)
    
    def decrypt_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt dictionary data"""
        json_str = self.decrypt(encrypted_data)
        return json.loads(json_str)


class APIKeyManager:
    """Manage API keys for authentication"""
    
    def __init__(self):
        self.keys = {}  # In production, use database
    
    def generate_api_key(self, user_id: str, name: str = "default") -> str:
        """Generate new API key"""
        api_key = secrets.token_urlsafe(32)
        key_hash = self._hash_key(api_key)
        
        self.keys[key_hash] = {
            'user_id': user_id,
            'name': name,
            'created_at': datetime.utcnow().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return associated data"""
        key_hash = self._hash_key(api_key)
        
        if key_hash in self.keys:
            # Update usage stats
            self.keys[key_hash]['last_used'] = datetime.utcnow().isoformat()
            self.keys[key_hash]['usage_count'] += 1
            return self.keys[key_hash]
        
        return None
    
    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        key_hash = self._hash_key(api_key)
        if key_hash in self.keys:
            del self.keys[key_hash]
            return True
        return False
    
    def _hash_key(self, api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()


class SecurityHeaders:
    """Security headers middleware"""
    
    HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
        'Content-Security-Policy': "default-src 'self'",
        'Referrer-Policy': 'strict-origin-when-cross-origin',
        'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
    }
    
    @classmethod
    def apply(cls, headers: Dict[str, str]) -> Dict[str, str]:
        """Apply security headers"""
        headers.update(cls.HEADERS)
        return headers


def rate_limit(max_requests: int = 60, window: int = 60):
    """Decorator for rate limiting functions"""
    limiter = RateLimiter(max_requests, window)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier (could be user_id, ip_address, etc.)
            identifier = kwargs.get('user_id', 'anonymous')
            
            if not limiter.is_allowed(identifier):
                reset_time = limiter.get_reset_time(identifier)
                raise RateLimitError(f"Rate limit exceeded. Try again in {reset_time} seconds")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_auth(token_manager: TokenManager):
    """Decorator to require authentication"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = kwargs.get('auth_token')
            if not token:
                raise AuthenticationError("Authentication required")
            
            try:
                payload = token_manager.verify_token(token)
                kwargs['auth_payload'] = payload
            except Exception as e:
                raise AuthenticationError(f"Authentication failed: {e}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


class SecureSession:
    """Secure session management"""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.sessions = {}  # In production, use Redis or database
    
    def create_session(self, user_id: str, data: Dict[str, Any]) -> str:
        """Create new session"""
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'data': data,
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': datetime.utcnow().isoformat()
        }
        
        # Encrypt sensitive session data
        encrypted_data = self.encryption_manager.encrypt_dict(session_data)
        self.sessions[session_id] = encrypted_data
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        if session_id not in self.sessions:
            return None
        
        try:
            encrypted_data = self.sessions[session_id]
            session_data = self.encryption_manager.decrypt_dict(encrypted_data)
            
            # Update last accessed time
            session_data['last_accessed'] = datetime.utcnow().isoformat()
            self.sessions[session_id] = self.encryption_manager.encrypt_dict(session_data)
            
            return session_data
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    def destroy_session(self, session_id: str) -> bool:
        """Destroy session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self, max_age_hours: int = 24):
        """Clean up expired sessions"""
        current_time = datetime.utcnow()
        expired = []
        
        for session_id, encrypted_data in self.sessions.items():
            try:
                session_data = self.encryption_manager.decrypt_dict(encrypted_data)
                last_accessed = datetime.fromisoformat(session_data['last_accessed'])
                
                if (current_time - last_accessed).total_seconds() > max_age_hours * 3600:
                    expired.append(session_id)
            except Exception as e:
                logger.error(f"Error checking session {session_id}: {e}")
                expired.append(session_id)
        
        for session_id in expired:
            del self.sessions[session_id]
        
        return len(expired)