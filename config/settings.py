import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(BASE_DIR / ".env")
# Reuse the dual Kaggle gateway configuration from the bundled PlanA reference
# when Contract_DL has not overridden those values in its own .env.
load_dotenv(BASE_DIR / "ai_extract" / "a3" / ".env")

# Make shared/ importable (especially inside Docker)
shared_path = str(BASE_DIR / "shared")
if shared_path not in sys.path:
    sys.path.insert(0, shared_path)

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.0/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = os.getenv("SECRET_KEY", "django-insecure-default-change-me")
# Keep document encryption independent of session/JWT signing-key rotation.
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', SECRET_KEY)

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.getenv("DEBUG", "True") == "True"

# Allow any host when deploying via tunnel or set via env
_env_hosts = [h.strip().strip("'\"") for h in os.getenv("ALLOWED_HOSTS", "*").split(",") if h.strip()]
ALLOWED_HOSTS = _env_hosts if _env_hosts else ["*"]
if "*" not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.extend([".trycloudflare.com", ".ngrok-free.app", ".ngrok-free.dev"])

# Trusted origins for CSRF protection when accessed over HTTPS tunnels
CSRF_TRUSTED_ORIGINS = [
    "https://*.trycloudflare.com",
    "https://*.ngrok-free.app",
    "https://*.ngrok-free.dev",
    "http://127.0.0.1",
    "http://localhost",
]



# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Custom Apps
    "contracts.apps.ContractsConfig",
    "ai_extract.apps.AiExtractConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "shared.jwt_middleware.JWTMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "shared.jwt_middleware.JWTUserMiddleware",
    "contracts.middleware.AutoDemoLoginMiddleware",
    "contracts.middleware.UserActionAuditMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("DB_NAME", "contract_db"),
        "USER": os.getenv("DB_USER", "postgres"),
        "PASSWORD": os.getenv("DB_PASSWORD", "postgrespassword"),
        "HOST": os.getenv("DB_HOST", "localhost"),
        "PORT": os.getenv("DB_PORT", "5432"),
        "OPTIONS": {
            "sslmode": os.getenv("DB_SSLMODE", "disable"),
            "connect_timeout": max(3, min(20, int(os.getenv("DB_CONNECT_TIMEOUT", "5")))),
        },
    }
}


# Password validation
FABRIC_GATEWAY_URL = os.getenv('FABRIC_GATEWAY_URL', 'http://fabric-gateway:5000')
FABRIC_GATEWAY_TOKEN = os.getenv('FABRIC_GATEWAY_TOKEN', '')
DATABASES['workflow'] = {**DATABASES['default'], 'NAME': os.getenv('WORKFLOW_DB_NAME', 'workflow_db')}

# https://docs.djangoproject.com/en/5.0/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.0/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.0/howto/static-files/

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = []

MEDIA_URL = "media/"
MEDIA_ROOT = BASE_DIR / "media"

# Default primary key field type
# https://docs.djangoproject.com/en/5.0/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# Custom Auth User Model
AUTH_USER_MODEL = "contracts.User"

# AI Inference Service Configuration
AI_SERVICE_URL = os.environ.get(
    "AI_SERVICE_URL", "http://localhost:8000"
)

# Kaggle Qwen AI Extract & Summarize Service
# Points to the kaggle-qwen-service Docker container (or ngrok URL when Kaggle notebook is running)
KAGGLE_QWEN_SERVICE_URL = os.environ.get(
    "KAGGLE_QWEN_SERVICE_URL",
    os.environ.get("KAGGLE_AI_URL", "http://kaggle-qwen-service:8000")
)

# PlanA remote ColPali endpoint (Kaggle/ngrok) and shared Qdrant storage.
COLPALI_BASE_URL = os.environ.get("COLPALI_BASE_URL", "")
COLPALI_ENABLED = os.environ.get('COLPALI_ENABLED', 'True').lower() == 'true'
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "")
A3_RELATED_PASSAGES_ENABLED = os.environ.get('A3_RELATED_PASSAGES_ENABLED','True').lower()=='true'
A3_RELATED_PASSAGE_CANDIDATES = int(os.environ.get('A3_RELATED_PASSAGE_CANDIDATES','5'))
A3_RELATED_PASSAGE_LIMIT = int(os.environ.get('A3_RELATED_PASSAGE_LIMIT','2'))
A3_RELATED_PASSAGE_MIN_LOGIT = float(os.environ.get('A3_RELATED_PASSAGE_MIN_LOGIT','0'))
QWEN_MODEL_ID = os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
# Output budgets do not change the serving engine's context window. Re-run
# the Kaggle model cell after changing QWEN_MAX_MODEL_LEN there.
QWEN_ANSWER_MAX_TOKENS = max(256, int(os.environ.get("QWEN_ANSWER_MAX_TOKENS", "8192")))
QWEN_ANSWER_MAX_CLAIMS = max(8, int(os.environ.get("QWEN_ANSWER_MAX_CLAIMS", "96")))
QWEN_REQUEST_TIMEOUT = max(30, int(os.environ.get("QWEN_REQUEST_TIMEOUT", "600")))
QWEN_VERIFY_TIMEOUT = max(30, int(os.environ.get("QWEN_VERIFY_TIMEOUT", "180")))
A2_ANSWER_SOURCE_LIMIT = max(4, int(os.environ.get("A2_ANSWER_SOURCE_LIMIT", "24")))
A2_BATCH_RERANK = os.environ.get('A2_BATCH_RERANK', 'True').lower() == 'true'
A2_RERANK_CANDIDATES = int(os.environ.get('A2_RERANK_CANDIDATES', '20'))
A2_MAP_WORKERS = int(os.environ.get('A2_MAP_WORKERS', '2'))
A2_MAP_REDUCE_ENABLED = os.environ.get('A2_MAP_REDUCE_ENABLED', 'False').lower() == 'true'
A2_INFERENCE_CACHE_ENABLED = False  # Enable only after cross-request caching is requested.
A2_INFERENCE_CACHE_VERSION = os.environ.get('A2_INFERENCE_CACHE_VERSION', '1')
A2_SOURCES_PER_DOCUMENT = max(2, int(os.environ.get("A2_SOURCES_PER_DOCUMENT", "6")))
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
RAG_INFERENCE_URL = os.environ.get("RAG_INFERENCE_URL", "")
SURYA_PAGE_READ_TIMEOUT = max(0, int(os.environ.get("SURYA_PAGE_READ_TIMEOUT", "0")))
A3_PAGE_CACHE_ROOT = BASE_DIR / 'scratch' / 'a3-page-cache'
RAG_LOCAL_MODELS = os.environ.get("RAG_LOCAL_MODELS", "False").lower() == "true"
A3_HYBRID_ENABLED = os.environ.get("A3_HYBRID_ENABLED", "True").lower() == "true"

# Workflow Service URL
WORKFLOW_SERVICE_URL = os.environ.get(
    "WORKFLOW_SERVICE_URL", "http://localhost:8003"
)

# Authentication Settings
LOGIN_URL = '/login/'



# Structure provider is independent of the Qwen answer-generation model.
A3_ENTITY_PROVIDER = os.environ.get('A3_ENTITY_PROVIDER','gliner')

# Source structure and typed entity mentions are separate inference stages.
A3_GRAPH_ENABLED = os.environ.get("A3_GRAPH_ENABLED", "True").lower() == "true"
A3_KNOWLEDGE_ENABLED = os.environ.get("A3_KNOWLEDGE_ENABLED", "True").lower() == "true"
NEO4J_URI = os.environ.get("NEO4J_URI", "")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
A3_VISUAL_NEAR_DISTANCE = float(os.environ.get("A3_VISUAL_NEAR_DISTANCE", "80"))

A2_SEARCH_WORKERS = int(os.environ.get("A2_SEARCH_WORKERS", "2"))

QWEN_TOKEN_PREFLIGHT = os.environ.get("QWEN_TOKEN_PREFLIGHT", "True").lower() == "true"
