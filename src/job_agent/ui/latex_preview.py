from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


_BLOCKED_COMMANDS = re.compile(
    r"\\(?:input|include|write18|openin|openout|read)(?![A-Za-z@])|"
    r"\\usepackage\s*\{shellesc\}",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LatexPreviewResult:
    success: bool
    pdf_path: Path | None = None
    png_path: Path | None = None
    message: str = ""
    diagnostic: str = ""
    log_path: Path | None = None


def _extract_latex_diagnostic(output: str) -> str:
    missing = re.search(
        r"(?:LaTeX Error:\s*)?File [`']([^`']+\.(?:sty|cls|tex))[`'] not found",
        output,
        re.IGNORECASE,
    )
    if missing:
        return (
            f"缺少模板依赖文件 {missing.group(1)}。当前只导入了单个 .tex 文件，"
            "请安装该宏包，或将模板依赖改为本机已有的标准宏包。"
        )
    if re.search(r"Undefined control sequence", output, re.IGNORECASE):
        line = re.search(r"l\.(\d+)", output)
        suffix = f"（约在第 {line.group(1)} 行）" if line else ""
        return f"存在无法识别的 LaTeX 命令{suffix}，请检查命令拼写或所需宏包。"
    if re.search(
        r"(fontspec error|font .* not found|cannot find.*font)",
        output,
        re.IGNORECASE,
    ):
        return "模板引用了本机未安装的字体，请更换字体或安装对应字体。"
    latex_error = re.search(r"!\s*(LaTeX Error:[^\r\n]+)", output)
    if latex_error:
        return latex_error.group(1).strip()
    if re.search(r"Emergency stop|Fatal error", output, re.IGNORECASE):
        return "编译器遇到致命错误，通常由模板依赖缺失或 LaTeX 语法不完整导致。"
    return "编译器未给出可归类的错误，请下载编译日志查看具体行号和上下文。"


def _find_pdftoppm() -> str | None:
    direct = shutil.which("pdftoppm.exe") or shutil.which("pdftoppm")
    if direct is None:
        return None
    direct_path = Path(direct)
    if direct_path.suffix.casefold() != ".cmd":
        return direct

    # Codex desktop exposes a lightweight command wrapper on Windows. Some
    # runtime versions point at the pre-relocation Poppler directory, so prefer
    # the bundled executable beside that runtime when it is present.
    if len(direct_path.parents) >= 3:
        bundled = (
            direct_path.parents[2]
            / "native"
            / "poppler"
            / "Library"
            / "bin"
            / "pdftoppm.exe"
        )
        if bundled.is_file():
            return str(bundled)
    return direct


def validate_latex_source(source: str) -> None:
    if _BLOCKED_COMMANDS.search(source):
        raise ValueError("LaTeX 包含不允许的外部文件或命令调用。")


def compile_latex_preview(
    source_path: Path,
    *,
    timeout_seconds: int = 45,
) -> LatexPreviewResult:
    source_path = Path(source_path).resolve()
    if source_path.suffix.casefold() != ".tex" or not source_path.is_file():
        return LatexPreviewResult(False, message="没有可编译的 LaTeX 源文件。")
    try:
        source = source_path.read_text(encoding="utf-8")
        validate_latex_source(source)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return LatexPreviewResult(False, message=str(exc))

    compiler = shutil.which("xelatex") or shutil.which("pdflatex")
    if compiler is None:
        return LatexPreviewResult(False, message="本机没有安装 XeLaTeX 或 PDFLaTeX。")

    output_dir = source_path.parent / ".internal" / "resume_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-no-shell-escape",
        f"-output-directory={output_dir}",
        str(source_path),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=source_path.parent,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return LatexPreviewResult(
            False,
            message="LaTeX 编译超时或无法启动。",
            diagnostic="请确认本机 LaTeX 编译器可正常运行，并检查模板是否需要联网安装宏包。",
        )
    pdf_path = output_dir / f"{source_path.stem}.pdf"
    if completed.returncode != 0 or not pdf_path.exists():
        compiler_log_path = output_dir / f"{source_path.stem}.log"
        compiler_output = "\n".join(
            part
            for part in (
                str(getattr(completed, "stdout", "") or ""),
                str(getattr(completed, "stderr", "") or ""),
                (
                    compiler_log_path.read_text(encoding="utf-8", errors="replace")
                    if compiler_log_path.is_file()
                    else ""
                ),
            )
            if part
        )
        if not compiler_log_path.is_file() and compiler_output:
            compiler_log_path.write_text(compiler_output, encoding="utf-8")
        return LatexPreviewResult(
            False,
            message="LaTeX 编译失败。",
            diagnostic=_extract_latex_diagnostic(compiler_output),
            log_path=compiler_log_path if compiler_log_path.is_file() else None,
        )

    png_path: Path | None = None
    renderer = _find_pdftoppm()
    if renderer is not None:
        prefix = output_dir / f"{source_path.stem}_preview"
        try:
            rendered = subprocess.run(
                [
                    renderer,
                    "-png",
                    "-f",
                    "1",
                    "-singlefile",
                    "-r",
                    "130",
                    str(pdf_path),
                    str(prefix),
                ],
                cwd=output_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
            candidate = prefix.with_suffix(".png")
            if rendered.returncode == 0 and candidate.exists():
                png_path = candidate
        except (OSError, subprocess.TimeoutExpired):
            png_path = None

    return LatexPreviewResult(
        True,
        pdf_path=pdf_path,
        png_path=png_path,
        message="LaTeX 简历已安全编译。",
        log_path=output_dir / f"{source_path.stem}.log",
    )
