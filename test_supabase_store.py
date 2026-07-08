import json

import supabase_store
import texgen as tg
from export_texture_gallery import build_index


def make_store(**kw):
    return supabase_store.SupabaseStorage(
        url="https://proj.supabase.co", service_key="svc", **kw)


def test_public_url_shape():
    s = make_store()
    assert s.public_url("img/abc_v0.png") == (
        "https://proj.supabase.co/storage/v1/object/public/textures/img/abc_v0.png")
    s2 = make_store(bucket="sprites")
    assert "/public/sprites/" in s2.public_url("x")


def test_upload_hits_object_endpoint_with_upsert(monkeypatch):
    calls = []
    s = make_store()
    monkeypatch.setattr(s, "_request",
                        lambda method, url, body=None, headers=None:
                        calls.append((method, url, body, headers)) or b"{}")
    url = s.upload_png("img/abc_v0.png", b"png")
    method, req_url, body, headers = calls[0]
    assert method == "POST"
    assert req_url.endswith("/storage/v1/object/textures/img/abc_v0.png")
    assert body == b"png"
    assert headers["x-upsert"] == "true"
    assert headers["Content-Type"] == "image/png"
    assert headers["Authorization"] == "Bearer svc"
    assert url == s.public_url("img/abc_v0.png")


def test_ensure_bucket_tolerates_already_exists(monkeypatch):
    s = make_store()

    def boom(method, url, body=None, headers=None):
        raise RuntimeError(f"POST {url} -> 409: already exists")

    monkeypatch.setattr(s, "_request", boom)
    s.ensure_bucket()  # must not raise


def test_missing_url_raises():
    try:
        supabase_store.SupabaseStorage(url="", service_key="k")
        assert False, "empty url should raise"
    except ValueError:
        pass


def test_writes_require_service_key():
    s = supabase_store.SupabaseStorage(url="https://p.supabase.co",
                                       service_key="")
    try:
        s._headers("image/png")
        assert False, "missing service key should raise"
    except ValueError:
        pass


def test_build_index_carries_base_url():
    idx = build_index([{"key": "k"}], "https://p.supabase.co/x/textures")
    assert idx["base_url"] == "https://p.supabase.co/x/textures"
    assert idx["count"] == 1
    assert "base_url" not in build_index([])


def test_decode_images_fetches_bucket_urls(monkeypatch):
    b = tg.RunPodComfyUIBackend(endpoint_id="ep", api_key="key")
    monkeypatch.setattr(b, "_fetch_url", lambda url: b"PNG:" + url.encode())
    out = b._decode_images(
        {"output": {"images": [
            {"filename": "a.png", "url": "https://proj.supabase.co/a.png"}]}}, 1)
    assert out == [b"PNG:https://proj.supabase.co/a.png"]
