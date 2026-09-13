"""
Dynamic Google Drive Catalog Sync Engine
Downloads, caches, and indexes tile textures from a shared Google Drive folder
using either Google Drive API OAuth or public downloads.
"""

import os
import io
import re
from pathlib import Path

SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
TOKEN_FILE = Path("token.json")
DRIVE_SYNC_DIR = Path("assets/materials/drive_sync")
DEFAULT_FOLDER_ID = "1_0c94oiLQP7YLU2BwGvpZwhuZxYnEjPI"

class DriveCatalogSync:
    def __init__(self, folder_url: str = f"https://drive.google.com/drive/folders/{DEFAULT_FOLDER_ID}"):
        self.folder_url = folder_url
        self.folder_id = DEFAULT_FOLDER_ID
        if "folders/" in folder_url:
            self.folder_id = folder_url.split("folders/")[1].split("?")[0]
        self.output_dir = DRIVE_SYNC_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_api_service(self):
        """Attempts to load credentials from token.json."""
        if TOKEN_FILE.exists():
            try:
                from google.oauth2.credentials import Credentials
                from google.auth.transport.requests import Request
                from googleapiclient.discovery import build
                creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
                if creds and creds.expired and creds.refresh_token:
                    creds.refresh(Request())
                if creds and creds.valid:
                    return build('drive', 'v3', credentials=creds)
            except Exception as e:
                print(f"[DriveSync] Token error: {e}")
        return None

    def sync_catalog(self):
        """Synchronizes tiles from Google Drive."""
        service = self.get_api_service()
        if service:
            try:
                from googleapiclient.http import MediaIoBaseDownload
                print(f"[DriveSync] Syncing with Google Drive API v3 for folder: {self.folder_id}")
                query = f"'{self.folder_id}' in parents and mimeType contains 'image/' and trashed = false"
                results = service.files().list(q=query, fields="files(id, name)").execute()
                files = results.get('files', [])
                for f in files:
                    fid = f['id']
                    fname = f['name']
                    dest = self.output_dir / fname
                    if not dest.exists():
                        req = service.files().get_media(fileId=fid)
                        fh = io.FileIO(str(dest), 'wb')
                        downloader = MediaIoBaseDownload(fh, req)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
            except Exception as api_err:
                print(f"[DriveSync] API sync error: {api_err}")
        else:
            try:
                import gdown
                gdown.download_folder(
                    url=self.folder_url,
                    output=str(self.output_dir),
                    quiet=True,
                    use_cookies=False,
                    remaining_ok=True
                )
            except Exception as gdown_err:
                print(f"[DriveSync] gdown sync info: {gdown_err}")

        return self.scan_local_catalog()

    def scan_local_catalog(self):
        """Scans the drive_sync directory and returns structured tile objects."""
        if not self.output_dir.exists():
            return []

        tiles = []
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

        idx = 1
        for f in sorted(self.output_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in valid_exts:
                stem = f.stem
                category = "marble" if "marble" in stem.lower() else ("wood" if "wood" in stem.lower() or "plank" in stem.lower() else "ceramic")
                surface_type = "wall" if "wall" in stem.lower() else "floor"
                finish = "glossy" if "glossy" in stem.lower() or "polish" in stem.lower() else "matte"
                
                if "whatsapp" in stem.lower():
                    name = f"Studio Catalogue Tile #{idx:02d}"
                else:
                    parts = stem.replace("-", "_").split("_")
                    name = " ".join([p.capitalize() for p in parts if not p.isdigit() and "x" not in p.lower() and p.lower() not in ("jpg", "png", "webp")])
                    if not name.strip():
                        name = stem.replace("_", " ").title()

                tile_id = f"drive-tile-{idx:02d}"
                rel_path = f"assets/materials/drive_sync/{f.name}"

                tiles.append({
                    "id": tile_id,
                    "name": name,
                    "brand": "Mentor Studio Collection",
                    "type": category,
                    "category": category,
                    "surface": "both",
                    "finish": finish,
                    "price": 125 + (idx % 6) * 10,
                    "img": rel_path,
                    "specs": "600x600 mm • Porcelain"
                })
                idx += 1

        return tiles
