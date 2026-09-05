"""
Google Drive API OAuth Authentication Utility
Run this script once to sign in with your Google account (with Viewer access).
It generates and saves 'token.json' locally, enabling continuous background sync.
"""

import os
import sys
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

# Read-only Drive Scope
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_FILE = Path("token.json")
CREDENTIALS_FILE = Path("credentials.json")
DEFAULT_FOLDER_ID = "1_0c94oiLQP7YLU2BwGvpZwhuZxYnEjPI"

def get_drive_service():
    """Authenticates and returns the Google Drive v3 service instance."""
    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception as e:
            print(f"[Auth] Existing token invalid: {e}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[Auth] Refreshing expired token...")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                print("\n" + "="*70)
                print("⚠️  MISSING 'credentials.json'")
                print("To connect directly via Google Drive API:")
                print("1. Go to Google Cloud Console (https://console.cloud.google.com/)")
                print("2. Create a Project -> Enable 'Google Drive API'")
                print("3. Go to 'Credentials' -> 'Create Credentials' -> 'OAuth Client ID' (Desktop App)")
                print("4. Download the JSON and save it as 'credentials.json' in this folder.")
                print("="*70 + "\n")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
        print(f"✅ Authorization successful! Saved {TOKEN_FILE}")

    return build('drive', 'v3', credentials=creds)

def sync_folder(folder_id=DEFAULT_FOLDER_ID, output_dir="assets/materials/drive_sync"):
    """Downloads all image files from the specified Drive folder."""
    service = get_drive_service()
    if not service:
        return []

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[Sync] Querying folder: {folder_id}...")
    query = f"'{folder_id}' in parents and mimeType contains 'image/' and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType, size)").execute()
    files = results.get('files', [])

    print(f"[Sync] Found {len(files)} image files in Drive folder.")
    synced_files = []

    for f in files:
        fid = f['id']
        fname = f['name']
        dest = out_path / fname

        if not dest.exists():
            print(f"[Sync] Downloading '{fname}'...")
            request = service.files().get_media(fileId=fid)
            fh = io.FileIO(str(dest), 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
        synced_files.append(str(dest))

    print(f"[OK] Sync complete! {len(synced_files)} textures available.")
    return synced_files

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER_ID
    sync_folder(folder)
