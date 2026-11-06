"""Print the user manual to one PDF, with a title page, the contents with page numbers,
page numbers on every page and bookmarks.

The PDF holds the pages of the documentation's User manual and Reference sections, in
the order ``docs/index.md`` gives them; links to the other pages (the roadmap, the
changelog...) go to the website. Sphinx builds the pages as one HTML page
(``singlehtml``), and Chrome prints it as a browser's Print button does. Chrome cannot
tell the page where a heading will land, so the page is printed twice: once with blanks
for the numbers, to find out where each heading landed, then with the numbers filled
in. The numbers take the room the blanks did, so nothing moves; the script checks
that, and prints again if something did.

    python docs/print_pdf.py acoustic-spaces-manual.pdf

It needs the ``docs`` extra (Sphinx, Furo, MyST and pypdf) and Chrome or Chromium:
``--chrome`` or the ``CHROME`` environment variable names the program (default
``google-chrome``).
"""

from __future__ import annotations

import argparse
import html
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, Fit, NameObject, TextStringObject

DOCS = Path(__file__).resolve().parent
# Where the website serves the pages the PDF leaves out.
SITE = "https://naelolaiz.github.io/acoustic-spaces/"
# The sections of docs/index.md the PDF prints; the others stay on the website.
PRINTED = ("User manual", "Reference")
BLANK = "0000"  # wide enough for any page number, so the real ones fit in its place
BREAK = '<div class="pdf-break"></div>'

TOCTREE = re.compile(r"```\{toctree\}\n(.*?)```", re.S)
DOC_LINK = re.compile(r"\]\(([A-Za-z0-9_-]+)\.md(#[^)\s]*)?\)")
DOCUMENT = re.compile(r'<span id="document-([^"]+)"></span>')
HEADING = re.compile(r'<h([234])>(.*?)<a class="headerlink" href="#([^"]+)"', re.S)
ID = re.compile(r'\bid="([^"]+)"')
HREF = re.compile(r'\bhref="#([^"]*)"')

STYLE = """
<style>
@page { size: A4; margin: 18mm 16mm;
  @bottom-center { content: counter(page); font-size: 9pt; color: #555; } }
@page :first { @bottom-center { content: none; } }
html { font-size: 90%; }
body { background: #fff; }
.pdf-break { break-before: page; }
article.pdf { max-width: none; padding: 0; }
.headerlink, .toctree-wrapper > p.caption { display: none; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; }
.highlight, pre, figure, img, table tr { break-inside: avoid; }
h2, h3, h4 { break-after: avoid; }
table.docutils { font-size: 0.85em; }
table.docutils td, table.docutils th { overflow-wrap: break-word; }
img { max-width: 100%; height: auto; }
.pdf-title-page { text-align: center; padding-top: 70mm; }
.pdf-title-page .title { font-size: 2.8em; font-weight: bold; margin: 0; }
.pdf-title-page .subtitle { font-size: 1.7em; margin: 0.4em 0 2.5em; }
.pdf-contents a:link, .pdf-contents a:visited, a.pdf-self { text-decoration: none;
  color: inherit; }
.pdf-contents .part { font-weight: bold; font-size: 1.15em; margin: 1.1em 0 0.3em;
  break-after: avoid; }
.pdf-contents .entry { display: flex; align-items: baseline; break-inside: avoid;
  margin: 0.1em 0; }
.pdf-contents .entry .dots { flex: 1; min-width: 1.5em; margin: 0 0.4em;
  border-bottom: 1px dotted currentColor; opacity: 0.5; }
.pdf-contents .entry .page { font-variant-numeric: tabular-nums; }
.pdf-contents .depth-0 { font-weight: 600; margin-top: 0.5em; }
.pdf-contents .depth-1 { padding-left: 1.5em; font-size: 0.95em; }
</style>
"""


@dataclass
class Heading:
    level: int  # 2 for a page's title (one below the PDF's own), 3 for its sections
    id: str
    title: str  # plain text


@dataclass
class Chapter:
    doc: str
    part: str
    headings: list[Heading] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.headings[0].id


def plain(fragment: str) -> str:
    """The text of an HTML fragment, without its tags."""
    return html.unescape(re.sub(r"<[^>]+>", "", fragment)).strip()


def parts(index: str) -> list[tuple[str, list[str]]]:
    """The printed sections of docs/index.md, each with its pages in order."""
    found = []
    for block in TOCTREE.findall(index):
        caption, docs = None, []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith(":caption:"):
                caption = line.removeprefix(":caption:").strip()
            elif line and not line.startswith(":") and line != "self":
                docs.append(line)
        if caption in PRINTED:
            found.append((caption, docs))
    if [caption for caption, _ in found] != list(PRINTED):
        sys.exit(f"docs/index.md has no toctree captioned {' and '.join(PRINTED)}")
    return found


def prepare(src: Path, printed: list[tuple[str, list[str]]]) -> None:
    """A copy of the docs that holds the printed pages alone, under a root page of
    its own, with their links to the other pages pointing at the website."""
    shutil.copytree(DOCS, src, ignore=shutil.ignore_patterns("_build", "__pycache__"))
    keep = {doc for _, docs in printed for doc in docs}
    for page in src.glob("*.md"):
        if page.stem not in keep:
            page.unlink()
            continue

        def to_site(m: re.Match) -> str:
            doc, anchor = m.group(1), m.group(2) or ""
            return m.group(0) if doc in keep else f"]({SITE}{doc}.html{anchor})"

        page.write_text(DOC_LINK.sub(to_site, page.read_text(encoding="utf-8")), "utf-8")
    trees = "\n\n".join(
        f"```{{toctree}}\n:caption: {caption}\n\n" + "\n".join(docs) + "\n```"
        for caption, docs in printed
    )
    (src / "print.md").write_text(f"# acoustic_spaces\n\n{trees}\n", encoding="utf-8")


def unique_ids(article: str) -> str:
    """One HTML page holds every page, so ids two pages share (two sections called
    Materials) would clash. Each id gets its page's name in front, and every link is
    turned to the id it means: the singlehtml builder writes a link to another page's
    section as #document-PAGE#section, and one to a page as #document-PAGE."""
    starts = [(m.start(), m.group(1)) for m in DOCUMENT.finditer(article)]
    pieces = [article[: starts[0][0]]]
    chunks = [
        (doc, article[start : starts[i + 1][0] if i + 1 < len(starts) else len(article)])
        for i, (start, doc) in enumerate(starts)
    ]
    owner: dict[str, list[str]] = {}
    first: dict[str, str] = {}
    for doc, chunk in chunks:
        ids = ID.findall(chunk)
        first[doc] = next(i for i in ids if not i.startswith("document-"))
        for i in ids:
            owner.setdefault(i, []).append(doc)

    def target(here: str, ref: str) -> str:
        if ref.startswith("document-"):
            doc, _, section = ref.removeprefix("document-").partition("#")
            return f"{doc}--{section or first.get(doc, '')}"
        docs = owner.get(ref, [])
        if here in docs:
            return f"{here}--{ref}"
        return f"{docs[0]}--{ref}" if len(docs) == 1 else ref

    for doc, chunk in chunks:
        chunk = ID.sub(lambda m, d=doc: f'id="{d}--{m.group(1)}"', chunk)
        chunk = HREF.sub(lambda m, d=doc: f'href="#{target(d, m.group(1))}"', chunk)
        pieces.append(chunk)
    return "".join(pieces)


def chapters_of(article: str, printed: list[tuple[str, list[str]]]) -> list[Chapter]:
    part_of = {doc: caption for caption, docs in printed for doc in docs}
    chapters = []
    for piece in re.split(r'(?=<span id="[^"]+--document-)', article):
        m = re.match(r'<span id="([^"]+)--document-', piece)
        if not m:
            continue
        chapter = Chapter(m.group(1), part_of[m.group(1)])
        for level, title, id in HEADING.findall(piece):
            chapter.headings.append(Heading(int(level), html.unescape(id), plain(title)))
        if not chapter.headings or chapter.headings[0].level != 2:
            sys.exit(f"{chapter.doc} has no title to point at")
        chapters.append(chapter)
    order = [doc for _, docs in printed for doc in docs]
    if [c.doc for c in chapters] != order:
        sys.exit(f"the printed page has {[c.doc for c in chapters]}, expected {order}")
    return chapters


def page_text(page: int | None) -> str:
    return BLANK if page is None else str(page + 1)


def contents_html(chapters: list[Chapter], pages: dict[str, int]) -> str:
    # The heading links to itself, which makes Chrome keep its place for a bookmark.
    lines = [
        '<section id="pdf-contents">',
        '<h2><a class="pdf-self" href="#pdf-contents">Contents</a></h2>',
        '<nav class="pdf-contents">',
    ]
    part = None
    for chapter in chapters:
        if chapter.part != part:
            part = chapter.part
            lines.append(f'<p class="part">{html.escape(part)}</p>')
        for heading in chapter.headings:
            if heading.level > 3:
                continue
            lines.append(
                f'<a class="entry depth-{heading.level - 2}" href="#{heading.id}">'
                f'<span class="name">{html.escape(heading.title)}</span>'
                '<span class="dots"></span>'
                f'<span class="page">{page_text(pages.get(heading.id))}</span></a>'
            )
    lines.append("</nav></section>")
    return "\n".join(lines)


def title_html(version: str) -> str:
    return (
        '<section class="pdf-title-page">'
        '<p class="title">acoustic_spaces</p>'
        '<p class="subtitle">User manual</p>'
        f"<p>Version {html.escape(version)}</p>"
        f'<p><a href="{SITE}">{SITE}</a></p>'
        "<p>acoustic_spaces contributors, GPL-3.0-or-later</p>"
        "</section>"
    )


def print_pdf(chrome: str, page: Path, out: Path) -> None:
    subprocess.run(
        [
            chrome,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=60000",
            f"--print-to-pdf={out}",
            page.as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if not out.exists() or out.stat().st_size == 0:
        sys.exit(f"Chrome printed nothing to {out}")


def destinations(reader: PdfReader) -> dict[str, tuple[int, object]]:
    """Every place a link in the PDF can go to, by its id in the HTML: its page (from
    0) and where on the page."""
    found = {}
    for name, dest in reader.named_destinations.items():
        id = html.unescape(name.lstrip("/"))
        found[id] = (reader.get_destination_page_number(dest), dest)
    return found


def add_bookmarks(
    writer: PdfWriter, chapters: list[Chapter], dests: dict[str, tuple[int, object]]
) -> None:
    def add(title: str, id: str, parent=None, open_: bool = False):
        page, dest = dests[id]
        top = dest.get("/Top")
        fit = Fit.xyz(left=dest.get("/Left"), top=top) if top is not None else Fit.fit()
        return writer.add_outline_item(title, page, parent=parent, fit=fit, is_open=open_)

    add("Contents", "pdf-contents")
    part, part_item = None, None
    for chapter in chapters:
        if chapter.part != part:
            part = chapter.part
            part_item = add(part, chapter.id, open_=True)
        item = add(chapter.headings[0].title, chapter.id, part_item)
        for heading in chapter.headings[1:]:
            if heading.level == 3 and heading.id in dests:
                add(heading.title, heading.id, item)


def point_links_at_pages(writer: PdfWriter) -> None:
    """Gives every link inside the PDF its page and place instead of a name to look
    up: Chrome writes links to named places, which not every PDF reader looks up."""
    names = writer.root_object.get("/Dests")
    if names is None:
        return
    names = names.get_object()
    for page in writer.pages:
        for annotation in page.get("/Annots", []):
            link = annotation.get_object()
            name = link.get("/Dest")
            if not isinstance(name, (NameObject, TextStringObject)):
                continue
            place = names.get(name)
            if place is None:
                continue
            place = place.get_object()
            if isinstance(place, DictionaryObject):
                place = place["/D"].get_object()
            link[NameObject("/Dest")] = ArrayObject(place)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path, help="the PDF to write")
    parser.add_argument("--chrome", default=os.environ.get("CHROME", "google-chrome"))
    args = parser.parse_args()
    # Chrome's PDFs make pypdf warn that they have more objects than they say; they
    # read fine.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    from acoustic_spaces import __version__

    printed = parts((DOCS / "index.md").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        src, built = Path(tmp) / "src", Path(tmp) / "html"
        prepare(src, printed)
        subprocess.run(
            [sys.executable, "-m", "sphinx", "-W", "--keep-going", "-q", "-b", "singlehtml"]
            + ["-D", "root_doc=print", str(src), str(built)],
            check=True,
        )
        source = (built / "print.html").read_text(encoding="utf-8")
        head = source[: source.index("</head>")]
        start = source.index('<article role="main"')
        start = source.index(">", start) + 1
        article = source[start : source.index("</article>", start)]
        # The root page's own title: the title page stands in its place.
        article = re.sub(r"<h1>.*?</h1>", "", article, count=1, flags=re.S)
        # Each page starts on a new sheet.
        article = unique_ids(DOCUMENT.sub(lambda m: BREAK + m.group(0), article))
        chapters = chapters_of(article, printed)
        needed = ["pdf-contents"] + [h.id for c in chapters for h in c.headings if h.level <= 3]

        def write(pages: dict[str, int], path: Path) -> None:
            path.write_text(
                head
                + STYLE
                + '</head><body data-theme="light"><article class="pdf">'
                + title_html(__version__)
                + BREAK
                + contents_html(chapters, pages)
                + article
                + "</article></body></html>",
                encoding="utf-8",
            )

        page_path, pdf_path = built / "print-pdf.html", Path(tmp) / "manual.pdf"
        pages: dict[str, int] = {}
        for _ in range(4):
            write(pages, page_path)
            print_pdf(args.chrome, page_path, pdf_path)
            reader = PdfReader(pdf_path)
            dests = destinations(reader)
            landed = {id: page for id, (page, _) in dests.items()}
            missing = [id for id in needed if id not in landed]
            if missing:
                sys.exit(f"no page for {len(missing)} places: {', '.join(missing[:10])}")
            if all(landed[id] == pages.get(id) for id in needed):
                break
            pages = landed
            print(f"printed {len(reader.pages)} pages, numbering them", file=sys.stderr)
        else:
            sys.exit("the page numbers keep moving")

        writer = PdfWriter(clone_from=reader)
        add_bookmarks(writer, chapters, dests)
        point_links_at_pages(writer)
        writer.page_mode = "/UseOutlines"
        writer.add_metadata({"/Title": "acoustic_spaces: User manual"})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "wb") as f:
            writer.write(f)
    print(f"{args.out}: {len(reader.pages)} pages", file=sys.stderr)


if __name__ == "__main__":
    main()
