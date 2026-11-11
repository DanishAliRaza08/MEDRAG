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


def test_admin_imports_pdf_guideline(client):
    import io

    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for page in range(3):
        c.drawString(72, 800, "Hospital Respiratory Protocol")  # running header, must be dropped
        lines = [["Bronchiectasis exacerbations should be treated with fourteen days of",
                  "antibiotics guided by previous sputum cultures. Contact Dr. Imran Ali on",
                  "0300-1234567 for advice. Airway clearance physiotherapy is recommended."],
                 ["Sputum should be sent for culture before antibiotics are started in every",
                  "exacerbation, and results reviewed at the follow-up appointment."],
                 ["Long-term macrolide therapy may be considered for frequent exacerbations",
                  "after specialist review and an electrocardiogram."]][page]
        for i, line in enumerate(lines):
            c.drawString(72, 760 - 14 * i, line)
        c.drawString(72, 40, f"Page {page + 1} of 3")
        c.showPage()
    c.save()
    ha, hc = login(client, "admin"), login(client)
    form = {"title": "Bronchiectasis protocol", "source": "local hospital, internal use", "modality": "xray"}
    files = {"file": ("protocol.pdf", buf.getvalue(), "application/pdf")}
    assert client.post("/api/kb/upload", data=form, files=files, headers=hc).status_code == 403
    r = client.post("/api/kb/upload", data=form, files=files, headers=ha)
    assert r.status_code == 200, r.text
    q = {"query": "bronchiectasis exacerbation antibiotics"}
    hit = client.post("/api/search", json=q, headers=hc).json()["hits"][0]
    assert hit["doc_id"] == r.json()["doc_id"]
    assert "0300" not in hit["text"] and "Hospital Respiratory Protocol" not in hit["text"]
    assert "fourteen days of antibiotics" in hit["text"]  # wrapped lines re-joined
    bad = client.post("/api/kb/upload", data=form, files={"file": ("x.exe", b"MZ", "application/octet-stream")},
                      headers=ha)
    assert bad.status_code == 400
