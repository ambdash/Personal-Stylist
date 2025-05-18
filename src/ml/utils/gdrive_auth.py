import json
import os
import argparse
from oauth2client.client import OAuth2WebServerFlow, flow_from_clientsecrets
from oauth2client.file import Storage
from oauth2client.tools import run_flow, argparser

def get_token_oauth2client(client_secret_file):
    """Generate Google Drive OAuth2 token."""
    # The scope for the Google Drive API
    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Create credentials directory if it doesn't exist
    creds_dir = os.path.dirname(client_secret_file)
    os.makedirs(creds_dir, exist_ok=True)

    # Load the client secrets from the JSON file
    flow = flow_from_clientsecrets(
        client_secret_file,
        scope=SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # This forces manual copy-paste of code
    )

    # Run the authentication flow and retrieve credentials
    storage = Storage(os.path.join(creds_dir, 'token_oauth2client.json'))
    
    # Use the default arguments from oauth2client.tools
    flags = argparser.parse_args(['--noauth_local_webserver'])
    credentials = run_flow(flow, storage, flags)

    token_path = os.path.join(creds_dir, 'generated_token.json')
    with open(token_path, 'w') as token_file:
        token_file.write(credentials.to_json())

    print(f"\nToken information saved to {token_path}")
    print("\nNow you can use this token with DVC. Run the following commands:")
    print("\ndvc remote modify myremote gdrive_use_service_account false")
    print("dvc remote modify myremote gdrive_client_id <your_client_id>")
    print("dvc remote modify myremote gdrive_client_secret <your_client_secret>")
    print(f"dvc remote modify myremote gdrive_user_credentials_file {token_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Google Drive OAuth2 token')
    parser.add_argument('client_secret', 
                       help='Path to client_secret.json file',
                       default='src/ml/utils/client_secret.json',
                       nargs='?')
    args = parser.parse_args()
    
    get_token_oauth2client(args.client_secret) 