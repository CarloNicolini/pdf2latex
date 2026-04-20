# PDF to LaTeX Converter

`pdf2latex.py` converts selected PDF pages into LaTeX snippets with OpenAI Vision and stitches them into a compilable `main.tex`.

It is designed for:

- extracting lecture notes, book chapters, and technical documents into editable LaTeX,
- building text+math-heavy documents (equations, tables, citations, section structure),
- recovering embedded raster graphics from pages into local PNG files,
- generating a bibliography file when references pages are provided explicitly,
- and running page conversion in parallel for faster throughput.

It is not intended to be a full LaTeX OCR replacement; it is a practical tool for drafting and editing workflows where human review is expected.

---

## Install (with Astral `uv`)

This project is easiest to run with `[uv](https://docs.astral.sh/uv/)`.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the project root:



`uv` reads the script dependencies declared in `pdf2latex.py` header:

```bash
python openai
python pymupdf
python typer
python python-dotenv
```

No manual `venv` setup is required to run commands with `uv run`.

---

## OpenAI credentials

The script expects `OPENAI_API_KEY` in the environment.

```bash
# Put this in .env (default load path)
OPENAI_API_KEY=sk-...
```

You can also pass a custom file:

```bash
uv run pdf2latex.py --env-file .env.custom mydoc.pdf
```

---

## Basic usage

From the script directory:

```bash
uv run pdf2latex.py mydoc.pdf
```

Output is written to:

- `./mydoc/mypdf_page_outputs...` directory (folder name based on input stem)
- `main.tex`
- `references.bib` (always created, even when empty)
- one `N.tex` for each processed page
- extracted images like `figure_p123_1.png`

Compile:

```bash
cd mydoc
tectonic main.tex
```

---

## Command-line options

### Core options

- `pdf_path` (positional): Source PDF file.
- `--page-range` (`1-3,8,10-12`): inclusive, comma-separated page selector.
- `--doc-class` (`article` default): LaTeX document class for generated `main.tex`.
- `--base-url`: use a custom OpenAI-compatible API base URL.
- `--env-file`: path to `.env` file.
- `--force / -f`: overwrite existing output directory after cleaning generated artifacts.

### Bibliography

- `--build-references` (default): attempt bibliography extraction from explicit pages.
- `--no-build-references`: disable all citation-aware behavior and convert citations to bracketed plain text.
- `--references-pages`: required only when `--build-references` is enabled; format supports single pages and ranges.

```bash
uv run pdf2latex.py --build-references --references-pages 1250-1280 book.pdf
uv run pdf2latex.py --no-build-references  book_excerpt.pdf --page-range 1-20
```

### Parallelism control

- `--batch-size` (default: `8`): number of pages processed in parallel per batch.

```bash
uv run pdf2latex.py --page-range 1-60 --batch-size 4 big-book.pdf
```

Larger values increase throughput but also increase API concurrency and local memory usage.

### Citation counter context

- `--start-section` and `--start-page` seed LaTeX counters when resuming mid-document.

---

## Example commands

```bash
# Single range, default settings
uv run pdf2latex.py --page-range 358-359 murphy_pbml_book_2_2022.pdf

# Mixed ranges
uv run pdf2latex.py --page-range "10-20,30,35-40,99" paper.pdf

# Disable bibliography extraction (safe for docs with many non-citation pages)
uv run pdf2latex.py --no-build-references --page-range 1-50 notes.pdf

# Process in smaller concurrent batches
uv run pdf2latex.py --batch-size 4 --page-range 1-120 dense_math.pdf

# Use explicit bibliography pages
uv run pdf2latex.py --build-references --references-pages 1180-1210 --page-range 1000-1210 monograph.pdf
```

---

## What the tool produces

For each input page, the model generates raw LaTeX body content only (no full preamble).

- `N.tex` files are included in `main.tex` in sorted page order.
- If figures are detected, files are exported to page-ordered names and referenced with `\\includegraphics{}`.
- If image extraction misses an item, a small `placeholder.png` is used as fallback.
- If references are built and citation markers are recognized, matching `\\cite{...}` entries are preserved.

---

## Known limitations

- **LLM quality varies**: complex tables, dense math, or unusual formatting can still produce imperfect LaTeX and may need manual fixes.
- **Reference extraction is controlled, not automatic**: the tool does not auto-detect bibliography pages in this mode; you must provide `--references-pages` when `--build-references` is enabled.
- **Image extraction is best-effort**: some vector graphics or non-raster visuals are approximated with `placeholder.png`.
- **No perfect deterministic output**: prompts are stochastic; identical runs can vary slightly unless you tune temperature and model choice.
- **API costs and quotas**: each page invokes vision/completion calls; token usage can be high on large documents.
- **Concurrency trade-off**: higher `--batch-size` speeds throughput but can hit API rate limits and increase transient memory/disk usage.

---

## Suggested workflow

1. Start with a small range (e.g., 1–3 pages) to check output quality.
2. Adjust `--doc-class`, model, and prompt-sensitive options if needed.
3. Convert larger ranges with suitable `--batch-size`.
4. Review generated `N.tex` and `main.tex`, then compile and patch manually where needed.

---

## Development

This script is intentionally pragmatic and script-oriented:

- uses `fitz`/PyMuPDF for extraction,
- uses OpenAI Vision for page understanding,
- and writes artifacts directly to local disk.

If you want deeper integration, you can add your own post-processing pass for:

- stricter math checks,
- citation normalization,
- template-specific LaTeX wrappers,
- or project-specific glossary/label conventions.

