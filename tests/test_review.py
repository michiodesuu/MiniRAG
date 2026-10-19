from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from urllib.parse import quote
import re
import threading

import pytest

from rag_ingest.models import Recognition
from rag_ingest.review import make_handler
from rag_ingest.stores import VisionStore

STEM = "doc_page_1_image_1"


@pytest.fixture
def server(paths, make_image):
    store = VisionStore(paths.method("vision"), paths)
    store.save(make_image(f"{STEM}.png"), Recognition("model text"))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), None)
    port = httpd.server_address[1]
    httpd.RequestHandlerClass = make_handler(store, port, "secret-token")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield store, port
    httpd.shutdown()


def request(port, method, path, host=None, body=None, headers=None):
    conn = HTTPConnection("127.0.0.1", port)
    conn.putrequest(method, path, skip_host=True)
    conn.putheader("Host", host or f"127.0.0.1:{port}")
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    data = (body or "").encode("utf-8")
    conn.putheader("Content-Length", str(len(data)))
    conn.endheaders(data)
    response = conn.getresponse()
    return response.status, response.read()


def save(port, **kw):
    return request(port, "POST", "/api/save/" + quote(f"{STEM}.json"), body="แก้แล้ว", **kw)


def test_page_embeds_token_and_saves_with_it(server):
    store, port = server
    status, page = request(port, "GET", "/")
    assert status == 200
    token = re.search(rb'name="review-token" content="([^"]+)"', page)[1].decode()
    assert save(port, headers={"X-Review-Token": token,
                            "Origin": f"http://127.0.0.1:{port}"})[0] == 200  # NOSONAR loopback test server
    assert store.load(STEM)["corrected_text"] == "แก้แล้ว"


@pytest.mark.parametrize("headers, host", [
    ({}, None),                                                  # cross-site form/fetch: no token
    ({"X-Review-Token": "wrong"}, None),
    ({"X-Review-Token": "secret-token", "Origin": "https://evil.example"}, None),
    ({"X-Review-Token": "secret-token"}, "evil.example:{port}"),  # DNS rebinding
])
def test_untrusted_saves_are_refused(server, headers, host):
    store, port = server
    status, _ = save(port, headers=headers, host=host and host.format(port=port))
    assert status == 403
    assert "corrected_text" not in store.load(STEM)


def test_rebinding_cannot_read(server):
    _, port = server
    assert request(port, "GET", "/api/items", host=f"evil.example:{port}")[0] == 403
    assert request(port, "GET", "/api/items", host=f"localhost:{port}")[0] == 200
