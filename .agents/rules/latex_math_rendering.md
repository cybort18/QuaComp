---
description: Rules for writing GitHub-compatible LaTeX math in Markdown files
always_on: true
---

# GitHub-Flavored Markdown (GFM) LaTeX Math Rendering Rules

To prevent raw unrendered LaTeX syntax errors on GitHub (KaTeX engine):

1. **NO Matrix Environments in Inline Math or List Items**:
   - Never write `\begin{pmatrix}`, `\begin{matrix}`, `\begin{bmatrix}` inside inline math (`$...$`) or list items.
   - GFM unescapes `\\` to `\` and converts `&` to `&amp;` inside lists, breaking KaTeX and leaving raw LaTeX text visible.
   - For quantum operators and channels, always use projector / Dirac outer-product notation:
     `$$E_0 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|, \quad E_1 = \sqrt{\gamma}|0\rangle\langle 1|$$`
     or diagonal notation `\text{diag}(...)`.

2. **Display Math Standalone Formatting**:
   - Multi-line or complex equations must use standalone `$$ ... $$` blocks placed on separate lines with empty lines above and below.
   - Avoid embedding complex environments inside indented bullet points.

3. **Markdown Tables**:
   - Never place LaTeX math (`$...$`) inside Markdown table cells.
   - Table column separators (`|`) conflict with Dirac kets/bras (`|0\rangle`) and conditional probabilities (`P(A|B)`).
   - Use standard Unicode symbols instead: `→`, `±`, `μs`, `×`, `≤`, `≥`, `√`.
