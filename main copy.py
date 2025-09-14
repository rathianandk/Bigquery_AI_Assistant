import os
import logging
from fastapi import FastAPI, HTTPException
import google.auth
import google.auth.transport.requests

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BigQuery Temporary Token Vending API",
    description="Provides short-lived access tokens for judges to run BigQuery queries.",
    version="1.0.0"
)

# --- Helper Function ---
def get_short_lived_token():
    """Generates a short-lived OAuth2 access token for the runtime service account."""
    try:
        # This will automatically use the service account credentials in a Cloud Run environment
        credentials, project_id = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        
        # Create a transport request object
        request = google.auth.transport.requests.Request()
        
        # Refresh the credentials to get an access token
        credentials.refresh(request)
        
        logger.info(f"Successfully generated a token for project: {project_id}")
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to generate token: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Could not obtain credentials. Ensure the service is running with a service account with adequate permissions. Error: {e}"
        )

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
async def health_check():
    """Simple health check to confirm the service is running."""
    return {"status": "ok"}

@app.get("/token", summary="Get Short-Lived Access Token")
async def get_token():
    """
    Provides a short-lived OAuth2 access token.
    
    This token can be used by the BigQuery client to authenticate requests
    on behalf of the service account running this API.
    """
    token = get_short_lived_token()
    return {"access_token": token}

# To run this locally for testing (requires application-default credentials):
# uvicorn main:app --reload
