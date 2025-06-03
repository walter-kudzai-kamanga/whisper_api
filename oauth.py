from authlib.integrations.starlette_client import OAuth
from starlette.config import Config
from starlette.middleware.sessions import SessionMiddleware
import os
from dotenv import load_dotenv

load_dotenv()

# OAuth2 configuration
config = Config('.env')
oauth = OAuth(config)

# Google OAuth2
oauth.register(
    name='google',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile'
    }
)

# Facebook OAuth2
oauth.register(
    name='facebook',
    api_base_url='https://graph.facebook.com/v12.0/',
    access_token_url='https://graph.facebook.com/v12.0/oauth/access_token',
    access_token_params=None,
    authorize_url='https://www.facebook.com/v12.0/dialog/oauth',
    authorize_params=None,
    api_base_url='https://graph.facebook.com/v12.0/',
    client_kwargs={
        'scope': 'email public_profile'
    }
)

# Apple OAuth2
oauth.register(
    name='apple',
    api_base_url='https://appleid.apple.com',
    access_token_url='https://appleid.apple.com/auth/token',
    authorize_url='https://appleid.apple.com/auth/authorize',
    client_kwargs={
        'scope': 'email name'
    }
) 