#!/usr/bin/env python3
"""Supabase Storage backend for the texture/sprite database.

One public bucket holds the whole generated-art database:

    <bucket>/index.json                  manifest (same schema as the local
                                         web/textures/index.json, plus base_url)
    <bucket>/img/<key_hash>_v<i>.png     one file per variation

Writers (the generation pipeline) authenticate with the service-role key via
the Storage REST API; readers (browser gallery, Godot client) need no key at
all — the bucket is public and files are served from:

    {SUPABASE_URL}/storage/v1/object/public/<bucket>/<path>

Env vars:
    SUPABASE_URL          https://<project-ref>.supabase.co        (required)
    SUPABASE_SERVICE_KEY  service_role key — writers only          (required to publish)
    SUPABASE_BUCKET       bucket name (default "textures")

The RunPod worker can additionally upload its raw outputs straight to the
same project via Supabase's S3-compatible endpoint (see TEXTURES.md); this
module is about the canonical, downscaled files the game actually loads.

CLI:
    python supabase_store.py check     # create bucket + roundtrip a test file
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BUCKET = "worldengine"


class SupabaseStorage:
    def __init__(self, url: str | None = None, service_key: str | None = None,
                 bucket: str | None = None, timeout: float = 60.0, retries: int = 2):
        self.url = (url or os.getenv("SUPABASE_URL") or "").rstrip("/")
        self.service_key = service_key or os.getenv("SUPABASE_SERVICE_KEY") or ""
        self.bucket = bucket or os.getenv("SUPABASE_BUCKET") or DEFAULT_BUCKET
        self.timeout = timeout
        self.retries = retries
        if not self.url:
            raise ValueError("SUPABASE_URL is required (https://<project-ref>.supabase.co)")

    # -- urls ------------------------------------------------------------------
    def public_url(self, path: str) -> str:
        """Public, no-auth URL for a file in the bucket (bucket must be public)."""
        return f"{self.url}/storage/v1/object/public/{self.bucket}/{path}"

    def _api(self, path: str) -> str:
        return f"{self.url}/storage/v1/{path}"

    def _headers(self, content_type: str | None = None, upsert: bool = False) -> dict:
        if not self.service_key:
            raise ValueError("SUPABASE_SERVICE_KEY is required for writes")
        h = {"Authorization": f"Bearer {self.service_key}", "apikey": self.service_key}
        if content_type:
            h["Content-Type"] = content_type
        if upsert:
            h["x-upsert"] = "true"
        return h

    def _request(self, method: str, url: str, body: bytes | None = None,
                 headers: dict | None = None) -> bytes:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers or {},
                                         method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:300]
                last_err = RuntimeError(f"{method} {url} -> {e.code}: {detail}")
                if e.code < 500:              # 4xx won't heal on retry
                    raise last_err from e
            except (TimeoutError, urllib.error.URLError) as e:
                last_err = e
            if attempt < self.retries:
                time.sleep(1.5 * (2 ** attempt))
        raise RuntimeError(f"{method} {url} failed after {self.retries + 1} attempts: {last_err}")

    # -- bucket ----------------------------------------------------------------
    def ensure_bucket(self) -> None:
        """Create the public bucket if it doesn't exist yet (idempotent)."""
        body = json.dumps({"id": self.bucket, "name": self.bucket,
                           "public": True}).encode()
        try:
            self._request("POST", self._api("bucket"), body,
                          self._headers("application/json"))
        except RuntimeError as e:
            if "409" not in str(e) and "already exists" not in str(e).lower():
                raise

    # -- objects ---------------------------------------------------------------
    def upload(self, path: str, data: bytes,
               content_type: str = "application/octet-stream") -> str:
        """Upload (upsert) one file; returns its public URL."""
        self._request("POST",
                      self._api(f"object/{self.bucket}/{path}"),
                      data, self._headers(content_type, upsert=True))
        return self.public_url(path)

    def upload_png(self, path: str, data: bytes) -> str:
        return self.upload(path, data, "image/png")

    def upload_json(self, path: str, obj: dict) -> str:
        return self.upload(path, (json.dumps(obj, indent=1) + "\n").encode(),
                           "application/json")

    def download(self, path: str) -> bytes:
        """Fetch a file through the public URL (no auth needed)."""
        return self._request("GET", self.public_url(path))

    def list(self, prefix: str = "") -> list:
        """List objects under a prefix (paged; returns every entry)."""
        out, offset = [], 0
        while True:
            body = json.dumps({"prefix": prefix, "limit": 1000,
                               "offset": offset}).encode()
            page = json.loads(self._request(
                "POST", self._api(f"object/list/{self.bucket}"),
                body, self._headers("application/json")))
            out.extend(page)
            if len(page) < 1000:
                return out
            offset += len(page)

    def delete(self, paths: list) -> None:
        body = json.dumps({"prefixes": paths}).encode()
        self._request("DELETE", self._api(f"object/{self.bucket}"),
                      body, self._headers("application/json"))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["check"])
    args = ap.parse_args()
    if args.command == "check":
        store = SupabaseStorage()
        print(f"project: {store.url}  bucket: {store.bucket}")
        store.ensure_bucket()
        print("bucket ok")
        url = store.upload("healthcheck.txt", b"texture pipeline was here",
                           "text/plain")
        print("uploaded:", url)
        back = store.download("healthcheck.txt")
        assert back == b"texture pipeline was here", "roundtrip mismatch"
        store.delete(["healthcheck.txt"])
        print("roundtrip OK — Supabase storage is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
