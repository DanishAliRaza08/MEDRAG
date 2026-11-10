from medrag.imaging import synthetic
from tests.conftest import login


def _study(client, h):
    pt = client.post("/api/patients", json={"patient_id": "MRN-1", "age": 61, "sex": "M"}, headers=h).json()
    png = synthetic.to_png_bytes(synthetic.chest_xray("consolidation", seed=1, burn_text=True))
    r = client.post("/api/studies", headers=h, data={"patient_ref": pt["patient_ref"], "modality": "xray",
                    "clinical_note": "Pt John Doe febrile"}, files={"file": ("x.png", png, "image/png")})
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_index(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "MedRAG" in client.get("/").text


def test_login_failure_and_missing_token(client):
    assert client.post("/api/auth/login", json={"username": "clinician", "password": "x"}).status_code == 401
    assert client.get("/api/patients").status_code == 401
    assert client.get("/api/patients", headers={"Authorization": "Bearer forged.token"}).status_code == 401


def test_full_clinical_workflow(client):
    h = login(client)
    st = _study(client, h)
    assert "John" not in st["clinical_note"]
    assert "image_key" not in client.get(f"/api/studies/{st['id']}", headers=h).json()
    img = client.get(f"/api/studies/{st['id']}/image", headers=h)
    assert img.headers["content-type"] == "image/png"
    rep = client.post(f"/api/studies/{st['id']}/analyze", json={"question": "fever, review CXR"}, headers=h).json()
    assert rep["status"] == "verified"
    pdf = client.get(f"/api/reports/{rep['id']}/pdf", headers=h)
    assert pdf.content.startswith(b"%PDF") and b"DRAFT - NOT FOR CLINICAL USE" in pdf.content
    signed = client.post(f"/api/reports/{rep['id']}/signoff", json={"comment": "ok"}, headers=h).json()
    assert signed["status"] == "signed_off"
    again = client.post(f"/api/reports/{rep['id']}/reject", json={"comment": "x"}, headers=h)
    assert again.status_code == 403
    assert b"DRAFT - NOT FOR CLINICAL USE" not in client.get(f"/api/reports/{rep['id']}/pdf", headers=h).content


def test_viewer_is_read_only(client):
    hc, hv = login(client), login(client, "viewer")
    st = _study(client, hc)
    assert client.get("/api/studies", headers=hv).status_code == 200
    assert client.post(f"/api/studies/{st['id']}/analyze", json={"question": "review"}, headers=hv).status_code == 403
    rep = client.post(f"/api/studies/{st['id']}/analyze", json={"question": "review"}, headers=hc).json()
    assert client.post(f"/api/reports/{rep['id']}/signoff", json={}, headers=hv).status_code == 403
    assert client.post("/api/patients", json={"patient_id": "x"}, headers=hv).status_code == 403


def test_admin_cannot_sign_off_but_can_audit(client):
    hc, ha = login(client), login(client, "admin")
    st = _study(client, hc)
    rep = client.post(f"/api/studies/{st['id']}/analyze", json={"question": "review"}, headers=hc).json()
    assert client.post(f"/api/reports/{rep['id']}/signoff", json={}, headers=ha).status_code == 403
    assert client.get("/api/audit", headers=hc).status_code == 403
    log = client.get("/api/audit", headers=ha).json()
    assert any(e["action"] == "permission_denied" for e in log)
    assert client.get("/api/audit/verify", headers=ha).json()["valid"] is True


def test_reject_needs_reason(client):
    h = login(client)
    st = _study(client, h)
    rep = client.post(f"/api/studies/{st['id']}/analyze", json={"question": "review"}, headers=h).json()
    assert client.post(f"/api/reports/{rep['id']}/reject", json={"comment": ""}, headers=h).status_code == 403
    r = client.post(f"/api/reports/{rep['id']}/reject", json={"comment": "laterality wrong"}, headers=h).json()
    assert r["status"] == "rejected"


def test_search_ask_and_kb(client):
    h, ha = login(client, "viewer"), login(client, "admin")
    r = client.post("/api/search", json={"query": "CURB-65", "strategy": "hybrid+rerank"}, headers=h).json()
    assert r["hits"][0]["doc_id"] == "kb-cap"
    a = client.post("/api/ask", json={"question": "how is pneumonia severity assessed"}, headers=h).json()
    assert a["citations"]
    doc = {"title": "New guideline", "text": "Hypersensitivity pneumonitis causes ground-glass change on CT.",
           "modality": ["xray"]}
    assert client.post("/api/kb/documents", json=doc, headers=h).status_code == 403
    assert client.post("/api/kb/documents", json=doc, headers=ha).json()["chunks"] == 1
    assert client.get("/api/kb", headers=h).json()["documents"] >= 16


def test_validation_errors(client):
    h = login(client)
    assert client.post("/api/studies/st_nope/analyze", json={"question": "review"}, headers=h).status_code == 404
    assert client.post("/api/studies/st_nope/analyze", json={"question": "x", "variant": "bad"},
                       headers=h).status_code == 422
    assert client.get("/api/reports/rp_nope", headers=h).status_code == 404


def test_samples_endpoint(client):
    h = login(client)
    assert "xray" in client.get("/api/samples", headers=h).json()
    r = client.get("/api/samples/fundus/glaucoma", headers=h)
    assert r.content.startswith(b"\x89PNG")
    assert client.get("/api/samples/fundus/unicorn", headers=h).status_code == 404
