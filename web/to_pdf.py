# web/to_pdf.py — render a report page (web/dist/<key>.html) to PDF.
#
# Google Drive stopped serving HTML as web pages in 2016, so an .html upload
# shows a download screen, not the report. A PDF previews cleanly in Drive and
# shares as a normal link — and the datasheet design is print-first, so the PDF
# looks the same as the page. Playwright loads the file, lets the JS run (score
# count-up, meters, decorative canvas settle at their final state), then prints.
#
# Usage: python -m web.to_pdf            (every web/dist/*.html)
#        python -m web.to_pdf vortexify  (one)

import sys
from pathlib import Path

DIST = Path(__file__).resolve().parent / "dist"


def to_pdf(html_path: Path) -> Path:
    from playwright.sync_api import sync_playwright

    pdf_path = html_path.with_suffix(".pdf")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_viewport_size({"width": 1100, "height": 1400})
        page.goto(html_path.resolve().as_uri())
        page.wait_for_timeout(1500)  # let the load animations settle
        page.emulate_media(media="screen")  # keep the on-screen look, not print CSS
        # One continuous sheet sized to the actual content — no mid-section page
        # breaks, identical to the web page.
        height = page.evaluate("Math.ceil(document.documentElement.scrollHeight)")
        page.pdf(
            path=str(pdf_path),
            print_background=True,
            prefer_css_page_size=False,
            width="1100px",
            height=f"{height + 4}px",
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )
        browser.close()
    return pdf_path


def main() -> None:
    keys = sys.argv[1:] or [p.stem for p in DIST.glob("*.html")]
    for key in keys:
        html = DIST / f"{key}.html"
        if not html.exists():
            print(f"skip {key}: no {html}")
            continue
        print("pdf ->", to_pdf(html))


if __name__ == "__main__":
    main()
