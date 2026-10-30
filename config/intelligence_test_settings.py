"""Isolated A2/A3 integration tests; never connects to the production database."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
A3_HYBRID_ENABLED = False
RAG_INFERENCE_URL = ''
SECRET_KEY = 'isolated-intelligence-tests-only'
DEBUG = False
ALLOWED_HOSTS = ['testserver', 'localhost']
INSTALLED_APPS = ['django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'django.contrib.messages',
                  'django.contrib.staticfiles', 'contracts', 'ai_extract']
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
AUTH_USER_MODEL = 'contracts.User'
ROOT_URLCONF = 'contracts.tests.knowledge_urls'
MIDDLEWARE = ['django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware']
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates',
              'APP_DIRS': True, 'OPTIONS': {'context_processors':
                ['django.template.context_processors.request',
                 'django.contrib.auth.context_processors.auth']}}]
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
STATIC_URL = '/static/'
USE_TZ = True
TIME_ZONE = 'Asia/Bangkok'
LOGIN_URL = '/login/'
VLLM_BASE_URL = ''
COLPALI_BASE_URL = ''
KAGGLE_QWEN_SERVICE_URL = ''
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
# Historical contract migrations contain PostgreSQL-only ALTER statements.
# Isolated SQLite tests create current tables; PostgreSQL migration validation
# remains a separate deployment check.
MIGRATION_MODULES = {'contracts': None, 'ai_extract': None}

# Isolated ingestion uses observed source layout; Qwen never chooses relations.
# Explicit model protocol fixtures pass their provider directly.
A3_GRAPH_ENABLED = False
A3_KNOWLEDGE_ENABLED = False
A3_RELATED_PASSAGES_ENABLED = False
