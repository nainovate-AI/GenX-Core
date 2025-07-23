# services/proxy_service/src/core/security.py
"""
Enterprise Security Manager
Following OWASP Top 10 and Zero Trust principles
"""
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import jwt
import redis.asyncio as redis
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import structlog

from .config import get_settings

settings = get_settings()
logger = structlog.get_logger()


class SecurityManager:
    """
    Comprehensive security management
    """
    
    def __init__(self):
        self.redis_client = None
        self.encryption_key = self._derive_encryption_key()
        self.fernet = Fernet(self.encryption_key)
        
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from secret"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=settings.SECRET_KEY[:16].encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(settings.SECRET_KEY.encode())
        )
        return key
    
    async def initialize(self):
        """Initialize Redis connection for blacklist"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
    
    def create_token(self, user_data: Dict[str, Any]) -> str:
        """
        Create secure JWT token
        """
        # Add security claims
        payload = {
            **user_data,
            "exp": datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow(),
            "nbf": datetime.utcnow(),  # Not before
            "iss": settings.APP_NAME,
            "jti": secrets.token_urlsafe(16),  # Unique token ID
            "type": "access"
        }
        
        return jwt.encode(
            payload,
            settings.SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM
        )
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify JWT token with additional security checks
        """
        try:
            # Decode token
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": True, "verify_nbf": True}
            )
            
            # Verify issuer
            if payload.get("iss") != settings.APP_NAME:
                raise jwt.InvalidTokenError("Invalid issuer")
            
            # Verify token type
            if payload.get("type") != "access":
                raise jwt.InvalidTokenError("Invalid token type")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise
        except jwt.InvalidTokenError:
            raise
        except Exception as e:
            logger.error("Token verification error", error=str(e))
            raise jwt.InvalidTokenError("Invalid token")
    
    async def is_token_blacklisted(self, token: str) -> bool:
        """Check if token is blacklisted"""
        if not self.redis_client:
            return False
        
        # Get token ID
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False}
            )
            jti = payload.get("jti")
            if not jti:
                return False
            
            # Check blacklist
            return await self.redis_client.exists(f"blacklist:token:{jti}") > 0
            
        except:
            return True
    
    async def blacklist_token(self, token: str):
        """Add token to blacklist"""
        if not self.redis_client:
            return
        
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            exp = payload.get("exp")
            
            if jti and exp:
                # Calculate TTL
                ttl = max(int(exp - datetime.utcnow().timestamp()), 0)
                
                # Add to blacklist
                await self.redis_client.setex(
                    f"blacklist:token:{jti}",
                    ttl,
                    "1"
                )
                
        except Exception as e:
            logger.error("Failed to blacklist token", error=str(e))
    
    async def has_permission(
        self, 
        user_id: str, 
        permission: str
    ) -> bool:
        """
        Check user permissions (simplified RBAC)
        In production, this would check against a proper RBAC system
        """
        # For now, all authenticated users can read metrics
        if permission == "metrics:read":
            return True
        
        # Admin permissions would be checked here
        # Example: return await self.check_user_role(user_id, "admin")
        
        return False
    
    async def log_access(
        self,
        user_id: str,
        resource: str,
        action: str,
        ip_address: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log access for audit trail
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "ip_address": ip_address,
            "request_id": request_id,
            "metadata": metadata
        }
        
        # Log to structured logger
        logger.info("Access log", **log_entry)
        
        # In production, also send to audit log system
        # await self.send_to_audit_system(log_entry)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password using PBKDF2"""
        salt = secrets.token_bytes(32)
        key = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # iterations
        )
        return f"{salt.hex()}${key.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            salt_hex, key_hex = hashed.split('$')
            salt = bytes.fromhex(salt_hex)
            key = bytes.fromhex(key_hex)
            
            new_key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            return hmac.compare_digest(new_key, key)
            
        except Exception:
            return False
    
    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and secret"""
        api_key = f"gx_{secrets.token_urlsafe(32)}"
        api_secret = secrets.token_urlsafe(64)
        
        # Return key and hashed secret
        return api_key, self.hash_password(api_secret)
    
    async def validate_api_key(
        self, 
        api_key: str, 
        api_secret: str
    ) -> Optional[Dict[str, Any]]:
        """Validate API key and secret"""
        # In production, lookup from database
        # For now, return mock data
        
        # Example validation:
        # stored_secret_hash = await self.get_api_key_secret(api_key)
        # if self.verify_password(api_secret, stored_secret_hash):
        #     return {"user_id": "api_user", "permissions": ["metrics:read"]}
        
        return None