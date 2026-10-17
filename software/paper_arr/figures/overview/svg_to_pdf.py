"""Render overview_paper.svg to the PDF that the paper includes (overview_paper_svg.pdf).

pdflatex cannot read SVG. The SVG is printed to a one-page PDF at its own size (6.3 x 3.4 in) by headless Edge, which uses
the installed Comic Sans MS and Segoe UI Symbol fonts; Ghostscript (MiKTeX's mgs) then turns the text into outlines, so the
figure carries no fonts at all (Edge embeds one symbol subset as a Type 3 font otherwise). The paper includes the result with
\\includegraphics[width=\\textwidth,trim=0 0.24bp 0.48bp 0,clip]{...}: Edge's page is 454.08 x 245.04 pt against the SVG's
453.6 x 244.8 pt, the excess being blank margin at the right and bottom. python svg_to_pdf.py"""
import os, re, subprocess, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
svg = open(os.path.join(HERE, "overview_paper.svg"), encoding="utf-8").read()
svg = re.sub(r"^<\?xml[^>]*>\s*", "", svg)
svg = re.sub(r"<!DOCTYPE[^>]*>\s*", "", svg, flags=re.S)
html = ("<!doctype html><html><head><meta charset='utf-8'><style>@page{size:6.3in 3.4in;margin:0}"
        "html,body{margin:0;padding:0;overflow:hidden;width:6.3in;height:3.4in}svg{display:block;width:6.3in;height:3.4in}"
        "</style></head><body>" + svg + "</body></html>")
tmp = tempfile.mkdtemp()
page, raw = os.path.join(tmp, "wrap.html"), os.path.join(tmp, "raw.pdf")
open(page, "w", encoding="utf-8").write(html)
subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                f"--print-to-pdf={raw}", "file:///" + page.replace("\\", "/")], check=True, capture_output=True)
out = os.path.join(HERE, "overview_paper_svg.pdf")
subprocess.run(["mgs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                "-dCompatibilityLevel=1.5", f"-sOutputFile={out}", raw], check=True)
print("written", out)
