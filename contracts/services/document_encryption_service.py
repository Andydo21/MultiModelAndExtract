import os
import hashlib
from django.conf import settings
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def _get_encryption_key():
    """Derive a 256-bit key from settings.ENCRYPTION_KEY or settings.SECRET_KEY."""
    raw_key = getattr(settings, "ENCRYPTION_KEY", None) or settings.SECRET_KEY
    # Hash the key using SHA-256 to ensure it is exactly 32 bytes (256 bits)
    return hashlib.sha256(raw_key.encode('utf-8')).digest()

def encrypt_pdf(data: bytes) -> bytes:
    """Encrypt data using AES-256-GCM. Returns nonce + ciphertext + tag."""
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext

def decrypt_pdf(encrypted_data: bytes) -> bytes:
    """Decrypt data using AES-256-GCM."""
    if len(encrypted_data) < 12:
        raise ValueError("Invalid encrypted data size")
    key = _get_encryption_key()
    aesgcm = AESGCM(key)
    nonce = encrypted_data[:12]
    ciphertext = encrypted_data[12:]
    return aesgcm.decrypt(nonce, ciphertext, None)
