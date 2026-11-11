"""Build a knowledge base from PubMed Central (PMC) open-access articles.

Only articles carrying a Creative Commons *Attribution* (CC BY) licence are
fetched, so the text may be stored and redistributed with attribution. Each
article becomes a markdown file with front matter recording PMCID, journal,
year, authors and licence, and that attribution is shown with every citation.

Uses NCBI E-utilities (esearch + efetch) and stays under NCBI's limit of
3 requests/second for clients without an API key.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
TOOL = "medrag"

# (topic id, modalities, search phrase). One phrase per finding the system can report.
# (topic id, modality, phrase that must appear in the title, optional extra title/abstract words).
TOPICS: list[tuple[str, str, str, str]] = [
    ("pneumonia", "xray", "community-acquired pneumonia", ""),
    ("effusion", "xray", "pleural effusion", ""),
    ("pneumothorax", "xray", "pneumothorax", ""),
    ("heart-failure", "xray", "heart failure", "diagnosis"),
    ("tuberculosis", "xray", "pulmonary tuberculosis", "diagnosis"),
    ("nodule", "xray", "pulmonary nodule", ""),
    ("cxr", "xray", "chest radiograph", "interpretation"),
    ("diabetic-retinopathy", "fundus", "diabetic retinopathy", ""),
    ("glaucoma", "fundus", "glaucoma", "diagnosis"),
    ("amd", "fundus", "macular degeneration", ""),
    ("hypertensive-retinopathy", "fundus", "hypertensive retinopathy", ""),
    ("breast-pathology", "histopathology", "breast cancer", "pathology"),
    ("colorectal-pathology", "histopathology", "colorectal cancer", "pathology"),
    ("digital-pathology", "histopathology", "computational pathology", ""),
]

# Curated selection (reviewed by title from the TOPICS searches plus targeted gap searches): clinical reviews and
# guidelines only, no AI-method papers, case reports or molecular studies. Pinned so the corpus is reproducible.
PINNED: list[tuple[str, str, str]] = [
    # chest radiograph
    ("4042014", "pneumonia", "xray"), ("7433705", "pneumonia", "xray"), ("6527561", "pneumonia", "xray"),
    ("9030780", "effusion", "xray"), ("5745967", "effusion", "xray"), ("13000056", "effusion", "xray"),
    ("12208270", "pneumothorax", "xray"), ("10509123", "pneumothorax", "xray"), ("12656181", "pneumothorax", "xray"),
    ("6860389", "heart-failure", "xray"), ("12842177", "heart-failure", "xray"), ("5579422", "cardiomegaly", "xray"),
    ("8870832", "tuberculosis", "xray"), ("8113854", "tuberculosis", "xray"),
    ("11868311", "nodule", "xray"), ("6335705", "nodule", "xray"), ("9954842", "cxr", "xray"),
    # retinal fundus
    ("5488240", "diabetic-retinopathy", "fundus"), ("10436763", "diabetic-retinopathy", "fundus"),
    ("7187969", "diabetic-retinopathy", "fundus"),
    ("8862623", "glaucoma", "fundus"), ("8873198", "glaucoma", "fundus"), ("5116372", "glaucoma", "fundus"),
    ("11508937", "amd", "fundus"), ("12339655", "amd", "fundus"), ("6474281", "amd", "fundus"),
    # histopathology
    ("10855371", "breast-pathology", "histopathology"), ("11313581", "breast-pathology", "histopathology"),
    ("12651393", "breast-pathology", "histopathology"),
    ("8448718", "colorectal-pathology", "histopathology"), ("10948573", "colorectal-pathology", "histopathology"),
]

REVIEW_WORDS = ("(review[title] OR guideline[title] OR guidelines[title] OR consensus[title] "
                "OR recommendations[title] OR update[title] OR overview[title] OR approach[title])")


@dataclass
class Article:
    pmcid: str
    title: str
    journal: str
    year: str
    authors: str
    license: str
    license_url: str
    abstract: str
    body: str

    @property
    def citation(self) -> str:
        return f"{self.authors}. {self.journal} ({self.year}). PMC{self.pmcid}. {self.license}"


def _get(url: str, params: dict, timeout: float = 60.0, retries: int = 3) -> bytes:
    q = urllib.parse.urlencode({**params, "tool": TOOL})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{url}?{q}", timeout=timeout) as r:  # noqa: S310 - fixed NCBI host
                data = r.read()
            time.sleep(0.4)  # stay below 3 requests / second
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def search(phrase: str, extra: str = "", n: int = 3, since: int = 2015) -> list[str]:
    term = f'"{phrase}"[title] AND {REVIEW_WORDS} AND "cc by license"[filter] AND {since}:3000[pdat] ' \
           'NOT "case report"[title] NOT protocol[title] NOT "systematic review"[title] NOT "meta-analysis"[title]'
    if extra:
        term += " AND " + " AND ".join(f"{w}[title/abstract]" for w in extra.split())
    data = json.loads(_get(EUTILS + "esearch.fcgi", {"db": "pmc", "term": term, "retmax": n, "retmode": "json",
                                                     "sort": "relevance"}))
    return data["esearchresult"]["idlist"]


def _text(el) -> str:
    if el is None:
        return ""
    # drop citations, tables, figures, formulas inside the element
    for tag in ("xref", "table-wrap", "fig", "disp-formula", "inline-formula", "table"):
        for child in el.findall(f".//{tag}"):
            tail = child.tail  # Element.clear() also wipes the tail, i.e. the text AFTER the element
            child.clear()
            child.tail = tail
    s = "".join(el.itertext())
    s = re.sub(r"\[\s*[\d,\s–-]*\]", "", s)  # leftover [1, 2-4]
    s = re.sub(r"\(\s*[,;\s–-]*\)", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*[;,]\s*([.;,])", r"\1", s)  # "cardiomegaly;." left behind by removed citations
    s = re.sub(r"\s+([.,;:])", r"\1", s)
    # superscript citation ranges ("obesity.1,3-5 In") leave ".,–" or ".,," behind once the numbers are gone
    s = re.sub(r"([.;:!?])(?:[,–—-]|\s[–—-])+(?=\s|$)", r"\1", s)
    s = re.sub(r"([a-z0-9%)])(?:,[–—-]|[–—-],|,,+)(?=\s)", r"\1", s)
    s = re.sub(r"(?<=\w)[,–—-]{2,}(?=\s)", ",", s)  # "levels,,–, while" -> "levels, while"
    return re.sub(r"\s{2,}", " ", s)


SKIP_SECTIONS = re.compile(r"(acknowledg|funding|conflict|competing|author contrib|abbreviation|data availab|"
                           r"supplementary|ethic|declaration|consent|references|search strategy|method)", re.I)


def parse(xml: bytes, pmcid: str) -> Article | None:
    root = ET.fromstring(xml)
    art = root.find(".//article")
    if art is None:
        return None
    meta = art.find("front/article-meta")
    title = _text(meta.find("title-group/article-title")) if meta is not None else ""
    journal = _text(art.find("front/journal-meta//journal-title"))
    year = ""
    for d in art.findall(".//pub-date"):
        y = d.findtext("year")
        if y:
            year = y
            break
    names = []
    for c in art.findall(".//contrib[@contrib-type='author']")[:3]:
        sn, gn = c.findtext("name/surname"), c.findtext("name/given-names")
        if sn:
            names.append(f"{sn} {gn[:1] if gn else ''}".strip())
    authors = ", ".join(names[:1]) + (" et al." if len(names) > 1 else "")
    lic = art.find(".//license")
    lic_url = ""
    if lic is not None:
        for k, v in lic.attrib.items():
            if k.endswith("href"):
                lic_url = v
        ali = lic.find("{http://www.niso.org/schemas/ali/1.0/}license_ref")
        if ali is not None and ali.text:
            lic_url = ali.text.strip()
    if "creativecommons.org/licenses/by/" not in lic_url:
        return None  # only plain CC BY (no NC/ND/SA) is accepted
    version = re.search(r"/by/(\d\.\d)", lic_url)
    license_name = f"CC BY {version.group(1) if version else ''}".strip()
    abstract = " ".join(_text(p) for p in art.findall("front/article-meta/abstract//p"))
    parts = []
    body = art.find("body")
    if body is None:
        return None
    for sec in body.findall("sec"):
        head = _text(sec.find("title"))
        if head and SKIP_SECTIONS.search(head):
            continue
        paras = [_text(p) for p in sec.findall(".//p")]
        paras = [p for p in paras if len(p.split()) >= 8]
        if paras:
            parts.append((f"## {head}\n" if head else "") + "\n".join(paras))
    if not parts:
        paras = [_text(p) for p in body.findall(".//p")]
        parts = [p for p in paras if len(p.split()) >= 8]
    return Article(pmcid, title, journal, year, authors, license_name, lic_url, abstract, "\n\n".join(parts))


def fetch(pmcid: str) -> Article | None:
    return parse(_get(EUTILS + "efetch.fcgi", {"db": "pmc", "id": pmcid}), pmcid)


def to_markdown(a: Article, modality: str, topic: str, max_words: int = 6000) -> str:
    words = a.body.split(" ")
    body = a.body if len(words) <= max_words else " ".join(words[:max_words]).rsplit(".", 1)[0] + "."
    clean = lambda s: re.sub(r"[\r\n:]+", " ", s).strip()  # noqa: E731
    return (f"---\nid: pmc-{a.pmcid}\ntitle: {clean(a.title)}\nsource: {clean(a.citation)}\nmodality: {modality}\n"
            f"topic: {topic}\nlicense: {a.license}\nlicense_url: {a.license_url}\n"
            f"url: https://pmc.ncbi.nlm.nih.gov/articles/PMC{a.pmcid}/\n---\n"
            f"# Abstract\n{a.abstract}\n\n{body}\n")


def build_pinned(out_dir: Path, pinned=PINNED, log=print) -> list[dict]:
    """Fetch the curated, pinned articles (reproducible corpus)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pmcid, topic, modality in pinned:
        try:
            a = fetch(pmcid)
        except Exception as e:  # pragma: no cover - network
            log(f"  skip PMC{pmcid}: {e}")
            continue
        if a is None:
            log(f"  skip PMC{pmcid}: not CC BY or no full text")
            continue
        (out_dir / f"pmc{pmcid}.md").write_text(to_markdown(a, modality, topic), encoding="utf-8")
        manifest.append({**asdict(a), "body": None, "abstract": None, "topic": topic, "modality": modality,
                         "words": len(a.body.split())})
        log(f"  [{topic}] PMC{pmcid}: {a.title[:80]} ({a.license}, {len(a.body.split())} words)")
    _write_index(out_dir, manifest)
    return manifest


def _write_index(out_dir: Path, manifest: list[dict]) -> None:
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (out_dir / "ATTRIBUTION.md").write_text(
        "# Attribution\n\nThe articles in this folder were retrieved from PubMed Central and are published under "
        "Creative Commons Attribution (CC BY) licences, which permit redistribution with attribution. Text was "
        "reduced to the abstract and narrative sections (references, tables, figures and administrative sections "
        "removed); long articles were truncated. No other changes were made.\n\n"
        + "\n".join(f"- PMC{m['pmcid']}: {m['title']}. {m['authors']}. *{m['journal']}* ({m['year']}). "
                    f"{m['license']} ({m['license_url']})" for m in manifest) + "\n")


def build(out_dir: Path, per_topic: int = 3, topics=TOPICS, log=print) -> list[dict]:
    """Discover ``per_topic`` CC BY review/guideline articles for each topic by search (not curated)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest, seen = [], set()
    for topic, modality, phrase, extra in topics:
        got = 0
        for pmcid in search(phrase, extra, n=per_topic * 3):
            if got >= per_topic or pmcid in seen:
                continue
            try:
                a = fetch(pmcid)
            except Exception as e:  # pragma: no cover - network
                log(f"  skip PMC{pmcid}: {e}")
                continue
            if a is None or len(a.body.split()) < 400:
                continue
            seen.add(pmcid)
            (out_dir / f"pmc{pmcid}.md").write_text(to_markdown(a, modality, topic), encoding="utf-8")
            manifest.append({**asdict(a), "body": None, "abstract": None, "topic": topic, "modality": modality,
                             "words": len(a.body.split())})
            got += 1
            log(f"  [{topic}] PMC{pmcid}: {a.title[:80]} ({a.license})")
    _write_index(out_dir, manifest)
    return manifest
