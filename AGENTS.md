# QuaComp Development Guidelines & Agent Directives

## 1. Documentation & LaTeX Math Rendering Rules (Strict GFM Compliance)

When writing or editing Markdown files (`README.md`, `PRD.md`, docs, etc.) that render on GitHub:

### A. Avoid Fragile LaTeX Matrices in Inline Text & Bullet Lists
- **NEVER** use 2D matrix environments (`\begin{pmatrix}`, `\begin{bmatrix}`, `\begin{matrix}`) inside inline math (`$...$`) or within bullet points / list items.
- *Root Cause*: GitHub's CommonMark parser unescapes `\\` to `\` and transforms `&` to `&amp;` inside list items before KaTeX executes, causing KaTeX parsing errors and leaving raw unrendered LaTeX visible.
- *Quantum Operators*: Represent quantum channels and operators using canonical outer-product (projector) notation or explicit diagonal notation:
  - Correct: `$$E_0 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|, \quad E_1 = \sqrt{\gamma}|0\rangle\langle 1|$$`
  - Correct: `$$\text{diag}(1, \sqrt{1-\gamma})$$`
  - Prohibited: `$E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix}$`

### B. Display Math Block Rules
- Use standalone `$$ ... $$` math blocks placed on their own lines with blank lines before and after.
- Do not indent multi-line LaTeX blocks inside bullet lists unless absolutely required and verified.
- Avoid unescaped special characters (`&`, `_`, `*`, `|`) inside inline math that might conflict with Markdown formatting.

### C. Markdown Tables: Avoid LaTeX Math
- **NEVER** use LaTeX math (`$...$`) inside Markdown table cells (`| ... |`).
- *Root Cause*: The pipe character `|` in Dirac notation (`|0\rangle`) or conditional probabilities (`P(A|B)`) conflicts with table column delimiters, and GitHub often skips KaTeX rendering inside table cells.
- **Always use native Unicode characters in tables**:
  - Arrows: `→`, `←`, `↔` (do NOT use `$\to$`)
  - Greek letters: `μ`, `γ`, `λ`, `σ`, `ρ`
  - Math symbols: `±`, `×`, `·`, `≤`, `≥`, `≈`, `√`

---

## 2. Code Quality & System Architecture

- Maintain 100% test coverage across all test suites (`pytest`).
- Preserve Graceful Fallback: All native accelerations (C++ extension, Apple Metal, NVIDIA CUDA) must fall back cleanly to CPU/Python if native libraries or GPUs are unavailable.
- Do not bump the version beyond `1.0.0` unless explicitly requested.
