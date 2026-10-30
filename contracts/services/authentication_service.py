"""JWT creation after Django has authenticated the account."""
from datetime import timedelta
from django.utils import timezone
from django.conf import settings
import jwt

def token_for(user):
    role = user.role.role_name if user.role else 'USER'
    now = timezone.now()
    token = jwt.encode(dict(user_id=user.id, username=user.username, role=role, exp=now + timedelta(days=7), iat=now), settings.SECRET_KEY, algorithm='HS256')
    return dict(token=token, user=dict(id=user.id, username=user.username, email=user.email, role=role))
