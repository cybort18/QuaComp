# QuaComp AI Assistant Rules (GEMINI.md)

## Documentation & GitHub Flavored Markdown (GFM) LaTeX Math Rules

To ensure 100% flawless LaTeX math rendering on GitHub:

1. **NO Matrix Environments inside Inline Math or Bullet Lists**:
   - Never write `\begin{pmatrix}`, `\begin{matrix}`, or `\begin{bmatrix}` inside inline math (`$...$`) or list items.
   - GFM unescapes `\\` to `\` and converts `&` to `&amp;` inside lists, breaking KaTeX parsing and leaving raw LaTeX text visible.
   - For quantum channels and Kraus operators, always use canonical projector / Dirac outer-product notation:
     $$E_0 = |0\rangle\langle 0| + \sqrt{1-\gamma}|1\rangle\langle 1|, \quad E_1 = \sqrt{\gamma}|0\rangle\langle 1|$$
     or diagonal notation $\text{diag}(...)$.

2. **Display Math Formatting**:
   - Always place `$$ ... $$` math blocks on standalone lines separated by blank lines above and below.
   - Do not indent display math blocks inside bullet lists without careful checking.

3. **Markdown Tables**:
   - Never place LaTeX math (`$...$`) inside Markdown table cells.
   - Table column separators (`|`) conflict with Dirac kets/bras (`|0\rangle`) and conditional probabilities (`P(A|B)`).
   - Use standard Unicode symbols instead: `→`, `±`, `μs`, `×`, `≤`, `≥`, `√`.
