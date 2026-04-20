# /// script
# requires-python = ">=3.10"
# dependencies =[
#     "pymupdf",
#     "openai",
#     "typer",
#     "python-dotenv",
# ]
# ///

import base64
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import typer
from dotenv import load_dotenv
from openai import OpenAI

app = typer.Typer(help="Convert PDF pages to LaTeX using OpenAI Vision models.")

# Approximate costs per token (Input, Output) in USD
COST_ESTIMATES = {
    "gpt-4.1": (0.000005, 0.000015),
    "gpt-4.1-mini": (0.00000015, 0.0000006),
}

def clean_latex(text: str) -> str:
    """Strip markdown code block wrappers from the model's response."""
    text = text.strip()
    if text.startswith("```latex"):
        text = text[8:]
    elif text.startswith("```bibtex"):
        text = text[9:]
    elif text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def sanitize_latex_output(text: str) -> str:
    r"""
    Fix common model output glitches that commonly break LaTeX compilation.

    In particular, replace `\\{x}` with `\\{x\\}` because that appears in
    set/brace literals and otherwise creates unmatched brace tokens.
    """
    # Keep the pattern narrow to avoid touching normal LaTeX commands.
    text = re.sub(r"\\\{([^{}\\]+?)\}", r"\\{\1\\}", text)
    return text


def neutralize_citations(text: str) -> str:
    """Convert citation commands to bracketed labels when citations are disabled."""
    citation_re = re.compile(r"\\cite\w*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^{}]+)\}")

    def _replace(match: re.Match[str]) -> str:
        cites = ", ".join(part.strip() for part in match.group(1).split(","))
        return f"[{cites}]"

    return citation_re.sub(_replace, text)

def generate_bibtex_entries_from_page(client: OpenAI, page_text: str) -> list[dict]:
    """Parse a full page of bibliography text into a list of BibTeX entries using a cheap model."""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a bibliography parser. Given raw text from a bibliography/references page, "
                        "extract every individual reference and convert it to BibTeX format. "
                        "Return a JSON object with a single key 'entries', whose value is a list of objects, "
                        "each with keys 'bibkey' (a short unique identifier, e.g. 'Murphy2022') and "
                        "'bibtex' (the full BibTeX entry string). "
                        "If a text snippet is too short or ambiguous to produce a real entry, skip it. "
                        "Never use placeholder values like 'Author' or 'Title of the book' — extract the real data. "
                        "IMPORTANT: You MUST escape any ampersand characters (&) as \\& inside the bibtex string "
                        "to prevent LaTeX compilation errors (e.g. journal={Statistics \\& Probability Letters})."
                    )
                },
                {"role": "user", "content": f"Parse these bibliography entries:\n\n{page_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        data = json.loads(response.choices[0].message.content)
        entries = data.get("entries", [])
        return [e for e in entries if "bibkey" in e and "bibtex" in e]
    except Exception as e:
        typer.secho(f"  Warning: Failed to parse bibliography page: {e}", fg=typer.colors.YELLOW)
        return []

def parse_page_range(range_str: str, max_pages: int) -> list[int]:
    """
    Parse strings like '1-3,5,8-10' into a sorted 0-indexed page list.
    Ranges are inclusive (1-3 -> 1,2,3).
    """
    if not range_str:
        return list(range(max_pages))
    
    pages = set()
    for part in range_str.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"Invalid empty page token in range '{range_str}'")

        if "-" in part:
            bounds = part.split("-")
            if len(bounds) != 2:
                raise ValueError(f"Invalid page range token '{part}'. Use forms like 1-3.")
            start_str, end_str = [value.strip() for value in bounds]
            if not start_str or not end_str:
                raise ValueError(f"Invalid page range token '{part}'. Empty start or end value.")
            start = int(start_str)
            end = int(end_str)
            if start < 1 or end < 1:
                raise ValueError("Page numbers must be 1-based positive integers.")
            if start > max_pages or end > max_pages:
                raise ValueError(f"Range '{part}' exceeds document length ({max_pages} pages).")
            if start > end:
                start, end = end, start
            pages.update(range(start - 1, end))
        else:
            page_num = int(part)
            if page_num < 1:
                raise ValueError("Page numbers must be 1-based positive integers.")
            if page_num > max_pages:
                raise ValueError(f"Page '{part}' exceeds document length ({max_pages} pages).")
            pages.add(page_num - 1)
    
    # Sort and return 0-indexed pages.
    return sorted(pages)


def clean_output_artifacts(out_dir: Path) -> None:
    """Remove previously generated artifacts so reruns with --force stay consistent."""
    # Keep only generated numeric page files and extracted figure files.
    page_tex_re = re.compile(r"^\d+\.tex$")
    figure_re = re.compile(r"^figure_p\d+_\d+\.png$")

    for path in out_dir.glob("*.tex"):
        if page_tex_re.match(path.name):
            path.unlink(missing_ok=True)

    for path in out_dir.glob("figure_p*.png"):
        if figure_re.match(path.name):
            path.unlink(missing_ok=True)

    for name in ("references.bib", "main.tex", "placeholder.png"):
        candidate = out_dir / name
        if candidate.exists():
            candidate.unlink()


def build_system_prompt(
    extracted_image_filenames: list[str],
    bibkey_to_bibtex: dict[str, str],
    build_references: bool,
    start_section: Optional[int],
    start_page: Optional[int],
) -> str:
    """Build the LLM prompt for a single page."""
    system_prompt = (
        "You are an expert LaTeX typesetter. Your sole task is to convert the provided document page image into "
        "clean, directly compilable LaTeX code that can be included inside a \\begin{document} body.\n\n"
        "AVAILABLE PACKAGES (already loaded — use freely):\n"
        "amsmath, amssymb, bm (for \\bm{} bold math), mathpazo, subcaption, graphicx, geometry, booktabs, hyperref.\n\n"
        "STRICT OUTPUT RULES — violating any of these will cause a compilation error:\n"
        "1. Output ONLY raw LaTeX syntax. Do NOT use markdown (no ``` fences, no ** bold, no # headings).\n"
        "2. Do NOT include a preamble, \\documentclass, \\usepackage, \\begin{document}, or \\end{document}.\n"
        "3. The & character is ONLY valid inside tabular, array, align, or similar math environments. "
        "   NEVER write & outside of one of these. If the page contains a table, ALWAYS wrap it in "
        "\\begin{tabular}{...} ... \\end{tabular}. Missing this will break compilation.\n"
        "4. Every \\begin{env} must have a matching \\end{env}.\n"
        "5. Mathematical expressions must be enclosed in $ ... $, \\( ... \\), \\[ ... \\], or an equation/align environment.\n"
        "6. Special LaTeX characters (%, $, #, _, ^, {, }, ~, &, \\) must be properly escaped when used as literal text.\n"
        "7. When writing set literals in math mode, always use paired escaped braces (example: `\\{a,b\\}`). "
        "Never emit `\\{c}`-style unpaired forms.\n"
        "8. SECTION NUMBERING: NEVER use the starred form \\section*, \\subsection*, \\chapter*, etc. "
        "   LaTeX will number sections automatically. NEVER hardcode a number into the section title "
        "(e.g., do NOT write \\section{4.2 Example} — write \\section{Example} instead).\n"
        "9. BIBLIOGRAPHY: If the page is a bibliography or references section, output the entries as plain "
        "\\bibitem{key} entries inside a \\begin{thebibliography}{99} ... \\end{thebibliography} block.\n"
        "   For inline \\cite{} commands, you MUST use the exact keys provided in the mappings below. "
        "If a mapping is not provided, use closest matching key from 'Known BibTeX keys' list. "
        "Never invent a key that is not in the known keys list.\n"
        "10. Preserve the content and structure of the original page as faithfully as possible."
    )

    placeholder_path = "placeholder.png"
    if extracted_image_filenames:
        filenames_str = ", ".join(f"'{f}'" for f in extracted_image_filenames)
        system_prompt += (
            f"\n\nWe have extracted {len(extracted_image_filenames)} raster images from this page. "
            f"Assuming a top-to-bottom, left-to-right reading order, their filenames are: {filenames_str}. "
            "Please use these exact filenames inside \\includegraphics{{}} commands where the figures/images appear in the document. "
            "If there are more figures (e.g. vector graphics) than extracted filenames, use "
            f"'{placeholder_path}' for the missing ones."
        )
    else:
        system_prompt += (
            "\n\nNo raster images were extracted from this page. If you see a figure, diagram, or image, "
            f"you MUST use '{placeholder_path}' inside the \\includegraphics{{}} command to prevent compilation errors."
        )

    if bibkey_to_bibtex:
        key_list = ", ".join(sorted(bibkey_to_bibtex.keys()))
        system_prompt += (
            f"\n\nKnown BibTeX keys available for \\cite{{}} commands: {key_list}. "
            "Whenever you see a citation marker (e.g. [Author20], [15], etc.), use the exact matching key from this list."
        )
    if not build_references:
        system_prompt += (
            "\n\nCitation generation is disabled. Do not use \\cite commands or BibTeX-style citation macros. "
            "Use plain bracketed citations like [Author20] or [1] as literal text instead."
        )

    if start_section is not None or start_page is not None:
        system_prompt += "\n\nContext for numbering:"
        if start_section is not None:
            system_prompt += f"\n- The extraction starts at section {start_section}."
        if start_page is not None:
            system_prompt += f"\n- The extraction starts at physical page {start_page}."
        system_prompt += "\nPrefer using standard LaTeX sectioning commands (e.g., \\section) rather than hardcoding numbers."

    return system_prompt


def prepare_page_payload(pdf_path: Path, out_dir: Path, p_num: int, dpi: int = 150) -> tuple[int, str, list[str]]:
    """Render one page and extract its embedded raster images."""
    doc = fitz.open(pdf_path)
    try:
        human_p_num = p_num + 1
        page = doc[p_num]
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        b64_str = base64.b64encode(img_bytes).decode("utf-8")

        image_list = page.get_image_info(xrefs=True)

        def image_sort_key(img_info: dict) -> tuple[float, float]:
            bbox = img_info.get("bbox")
            if not bbox or len(bbox) < 4:
                return (0.0, 0.0)
            return (bbox[1], bbox[0])

        image_list.sort(key=image_sort_key)
        extracted_image_filenames: list[str] = []
        img_idx = 1
        for img in image_list:
            xref = img.get("xref")
            if not xref:
                continue

            img_pix = fitz.Pixmap(doc, xref)
            if img_pix.n - img_pix.alpha >= 4:
                img_pix = fitz.Pixmap(fitz.csRGB, img_pix)

            img_filename = f"figure_p{human_p_num}_{img_idx}.png"
            img_path = out_dir / img_filename
            img_pix.save(str(img_path))
            extracted_image_filenames.append(img_filename)
            img_idx += 1

        return human_p_num, b64_str, extracted_image_filenames
    finally:
        doc.close()

@app.command()
def convert(
    pdf_path: Path = typer.Argument(..., help="Path to the source PDF file", exists=True, dir_okay=False),
    env_file: Optional[Path] = typer.Option(None, help="Path to .env file for OPENAI_API_KEY"),
    model: str = typer.Option("gpt-4.1", help="OpenAI vision-compatible model"),
    page_range: Optional[str] = typer.Option(None, help="Page range (e.g., 1-5,8,11-13). Defaults to all pages."),
    build_references: bool = typer.Option(
        False,
        "--build-references/--no-build-references",
        help="Whether to extract and build the bibliography in BibTeX format",
    ),
    references_pages: Optional[str] = typer.Option(None, help="Page range for the bibliography (e.g., 1269-1288). Required if build-references is True."),
    doc_class: str = typer.Option("article", help="LaTeX documentclass for the main file"),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite if output directory already exists"),
    base_url: Optional[str] = typer.Option(None, help="Custom OpenAI-compatible base URL"),
    batch_size: int = typer.Option(8, help="Number of pages to process in parallel per batch"),
    start_section: Optional[int] = typer.Option(None, help="Starting section number for LaTeX counters"),
    start_page: Optional[int] = typer.Option(None, help="Starting page number for LaTeX counters"),
):
    """
    Extracts pages from a PDF, converts them to base64 images, and uses an OpenAI-compatible
    API to generate raw LaTeX code and BibTeX for each page.
    """
    if env_file and env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Look for default .env

    if start_section is not None and start_section < 1:
        typer.secho("Invalid --start-section: must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if start_page is not None and start_page < 1:
        typer.secho("Invalid --start-page: must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(1)
    if batch_size < 1:
        typer.secho("Invalid --batch-size: must be >= 1.", fg=typer.colors.RED)
        raise typer.Exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        typer.secho("Error: OPENAI_API_KEY not found in environment or .env file.", fg=typer.colors.RED)
        raise typer.Exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Handle directory creation
    out_dir = pdf_path.parent / pdf_path.stem
    if out_dir.exists():
        if not force:
            typer.confirm(f"Directory '{out_dir}' already exists. Modify its contents?", abort=True)
        elif not out_dir.is_dir():
            typer.secho(f"Output path '{out_dir}' exists but is not a directory.", fg=typer.colors.RED)
            raise typer.Exit(1)
        clean_output_artifacts(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    typer.secho(f"Opening {pdf_path.name}...", fg=typer.colors.BLUE)
    processed_pages: list[int] = []
    doc: Optional[fitz.Document] = None
    bibkey_to_bibtex: dict[str, str] = {}

    try:
        doc = fitz.open(pdf_path)
        try:
            pages_to_process = parse_page_range(page_range, len(doc))
        except ValueError as e:
            typer.secho(f"Invalid --page-range value: {e}", fg=typer.colors.RED)
            raise typer.Exit(1)

        if not pages_to_process:
            typer.secho("No valid pages found to process.", fg=typer.colors.RED)
            raise typer.Exit(1)

        # --- Phase 0: Pre-scan bibliography pages and build the registry ---
        if build_references:
            if references_pages:
                typer.secho("Scanning provided bibliography pages...", fg=typer.colors.BLUE)
                try:
                    bib_page_nums = parse_page_range(references_pages, len(doc))
                except ValueError as e:
                    typer.secho(f"Invalid --references-pages value: {e}", fg=typer.colors.RED)
                    raise typer.Exit(1)
                if bib_page_nums:
                    typer.secho(f"  Parsing {len(bib_page_nums)} bibliography page(s): "
                                f"{sorted(p + 1 for p in bib_page_nums)}", fg=typer.colors.MAGENTA)
                    for bib_p in sorted(bib_page_nums):
                        bib_page = doc[bib_p]
                        page_text = bib_page.get_text("text").strip()
                        if not page_text or len(page_text) < 50:
                            continue
                        typer.secho(f"  Parsing bibliography page {bib_p + 1}...", fg=typer.colors.MAGENTA)
                        entries = generate_bibtex_entries_from_page(client, page_text)
                        if entries:
                            typer.secho(f"    Extracted {len(entries)} entries.", fg=typer.colors.MAGENTA)
                            for entry in entries:
                                bibkey = entry["bibkey"]
                                if bibkey not in bibkey_to_bibtex:
                                    bibkey_to_bibtex[bibkey] = entry["bibtex"]
            else:
                typer.secho("  --build-references is True but no --references-pages provided. Bibliography extraction skipped.", fg=typer.colors.YELLOW)
        else:
            typer.secho("  Bibliography extraction disabled (--no-build-references).", fg=typer.colors.YELLOW)
            if references_pages:
                typer.secho("  --references-pages is ignored because --no-build-references is active.", fg=typer.colors.YELLOW)
    finally:
        if doc is not None:
            doc.close()

    placeholder_path = out_dir / "placeholder.png"
    if not placeholder_path.exists():
        placeholder_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="))

    async def process_page(p_num: int) -> dict:
        human_p_num = p_num + 1
        typer.secho(f"Processing page {human_p_num}...", fg=typer.colors.CYAN)
        try:
            _prepared_human, b64_str, extracted_image_filenames = await asyncio.to_thread(
                prepare_page_payload,
                pdf_path,
                out_dir,
                p_num,
            )
        except Exception as e:
            typer.secho(f"  Warning: Failed to prepare page {human_p_num}: {e}", fg=typer.colors.YELLOW)
            return {
                "page": human_p_num,
                "success": False,
            }

        system_prompt = build_system_prompt(
            extracted_image_filenames=extracted_image_filenames,
            bibkey_to_bibtex=bibkey_to_bibtex,
            build_references=build_references,
            start_section=start_section,
            start_page=start_page,
        )

        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_str}"}}]},
                ],
                temperature=0.1,
            )
        except Exception as e:
            typer.secho(f"API Error on page {human_p_num}: {e}", fg=typer.colors.RED)
            return {
                "page": human_p_num,
                "success": False,
            }

        usage = response.usage
        p_tokens = c_tokens = t_tokens = 0
        if usage:
            p_tokens = usage.prompt_tokens
            c_tokens = usage.completion_tokens
            t_tokens = usage.total_tokens
            typer.echo(f"  Tokens for page {human_p_num}: Prompt: {p_tokens} | Completion: {c_tokens} | Total: {t_tokens}")

        if not response.choices:
            typer.secho(f"  Warning: Empty completion for page {human_p_num}, skipping.", fg=typer.colors.YELLOW)
            return {"page": human_p_num, "success": False}

        choice = response.choices[0]
        if choice.message is None or choice.message.content is None:
            typer.secho(f"  Warning: Empty message content for page {human_p_num}, skipping.", fg=typer.colors.YELLOW)
            return {"page": human_p_num, "success": False}

        response_text = choice.message.content
        latex_content = clean_latex(response_text)
        latex_content = sanitize_latex_output(latex_content)
        contains_citation = build_references and ("\\cite{" in latex_content or "\\citep{" in latex_content or "\\citet{" in latex_content)
        if not build_references:
            latex_content = neutralize_citations(latex_content)

        return {
            "page": human_p_num,
            "success": True,
            "content": latex_content,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": t_tokens,
            "contains_citation": contains_citation,
        }

    async def process_pages_in_batches() -> list[dict]:
        results: list[dict] = []
        for idx in range(0, len(pages_to_process), batch_size):
            batch = pages_to_process[idx: idx + batch_size]
            batch_results = await asyncio.gather(*(process_page(p) for p in batch))
            for result in batch_results:
                if result is not None:
                    results.append(result)
        return results

    page_results = asyncio.run(process_pages_in_batches())
    successful_results = [result for result in page_results if result["success"]]
    successful_results.sort(key=lambda item: item["page"])

    if not successful_results:
        typer.secho("Warning: No pages were successfully processed.", fg=typer.colors.YELLOW)

    total_prompt_tokens = sum(result.get("prompt_tokens", 0) for result in successful_results)
    total_completion_tokens = sum(result.get("completion_tokens", 0) for result in successful_results)
    total_tokens = sum(result.get("total_tokens", 0) for result in successful_results)
    contains_citations = any(result.get("contains_citation", False) for result in successful_results)

    for result in successful_results:
        processed_pages.append(result["page"])
        out_file = out_dir / f"{result['page']}.tex"
        out_file.write_text(result["content"], encoding="utf-8")

    # Cost Calculation
    cost_in, cost_out = COST_ESTIMATES.get(model, (0.0, 0.0))
    total_cost = (total_prompt_tokens * cost_in) + (total_completion_tokens * cost_out)

    typer.secho("\n--- Conversion Summary ---", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"Total Pages Processed: {len(processed_pages)}")
    typer.echo(f"Total Prompt Tokens:     {total_prompt_tokens}")
    typer.echo(f"Total Completion Tokens: {total_completion_tokens}")
    typer.echo(f"Sum Total Tokens:        {total_tokens}")
    if total_cost > 0:
        typer.echo(f"Estimated Cost (USD):    ${total_cost:.4f}")
    else:
        typer.echo("Estimated Cost (USD):    Model pricing unknown. Check OpenAI pricing page.")

    # Write references.bib — always create so \bibliography{references} never errors
    bib_file_path = out_dir / "references.bib"
    unique_entries = list(bibkey_to_bibtex.values())
    if unique_entries:
        bib_file_path.write_text("\n\n".join(unique_entries), encoding="utf-8")
        typer.secho(f"Saved {len(unique_entries)} unique BibTeX entries to references.bib", fg=typer.colors.GREEN)
    else:
        bib_file_path.write_text("% No references extracted\n", encoding="utf-8")
        typer.secho("Created empty references.bib placeholder.", fg=typer.colors.YELLOW)

    # Create compilable main.tex
    main_tex_path = out_dir / "main.tex"
    preamble = f"""\\documentclass{{{doc_class}}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{amsmath, amssymb}}
\\usepackage{{bm}}
\\usepackage{{mathpazo}} % Palatino font as requested
\\usepackage{{subcaption}}
\\usepackage{{graphicx}}
\\usepackage{{geometry}}
\\usepackage{{booktabs}}
\\usepackage{{hyperref}}

\\begin{{document}}

"""
    if start_page is not None:
        preamble += f"\\setcounter{{page}}{{{start_page}}}\n"
    if start_section is not None:
        preamble += f"\\setcounter{{section}}{{{start_section - 1}}}\n"
        
    for p in processed_pages:
        preamble += f"\\input{{{p}.tex}}\n\\clearpage\n"
    
    include_bibliography = build_references and (contains_citations or bool(bibkey_to_bibtex))
    if include_bibliography:
        preamble += "\\bibliographystyle{plain}\n"
        preamble += "\\bibliography{references}\n"
        
    preamble += "\n\\end{document}\n"
    main_tex_path.write_text(preamble, encoding="utf-8")
    
    typer.secho(f"\nSuccess! All files saved in: {out_dir}", fg=typer.colors.GREEN)
    typer.secho(f"Compile the full document using: pdflatex {main_tex_path.name} (inside the folder)", fg=typer.colors.YELLOW)

if __name__ == "__main__":
    app()
