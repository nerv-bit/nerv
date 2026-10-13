// More aggressive: rewrite EVERY occurrence of `&mut rng` inside `mod tests`
// with `r`, after wrapping the line with `{ let r = &mut rng; ... }`.
//
// For multi-line statements (the typical `let x = ...,\n     ...;` form),
// this script wraps each individual call site with its own shadow by
// detecting the start of the let binding and the closing `;`.

const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/executor.rs";

let src = fs.readFileSync(path, "utf8");
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");

// Find `mod tests {` start.
let testStart = -1;
for (let i = 0; i < lines.length; i++) {
  if (lines[i].trim() === "mod tests {") {
    testStart = i;
    break;
  }
}
if (testStart < 0) {
  console.error("Could not find `mod tests {`.");
  process.exit(2);
}

// Find the closing `}` of mod tests (we trust there is one matching `}`).
// Easier: walk until we find a line that's exactly `}` and the line before was
// not more indented. The mod-tests block ends at the file's last `}`.
// We can detect by tracking brace depth, but since mod tests is the last
// nested block, just walk until we hit the end-of-file. We'll know the inner
// code wraps anything inside.

// Process: any line containing `&mut rng` gets rewritten to use `r` instead,
// with a shadow block. We do this for each line independently.
//
// If a line is already wrapped (has `let r = &mut rng`), skip it.
let fixCount = 0;
let inLetBinding = false; // are we inside a multi-line let?

for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  if (!line.includes("&mut rng")) continue;
  if (line.includes("let r = &mut rng")) continue;

  // Skip lines that are entirely comments.
  const trimmed = line.trimStart();
  if (trimmed.startsWith("//")) continue;

  // Replace `&mut rng` with `r` in the line and wrap with `{ let r = &mut rng; ... }`.
  // For simplicity, wrap each line individually.

  // But we also need to be careful: if the line is part of a multi-line expression,
  // wrapping with `{ ... }` could break things. The brief's proven pattern was:
  //   let (b, txid, nf) = { let r = &mut rng; one_leg_block(r, ..., vec![{ let __nf = h(r); __nf }], ...); };
  // So they did wrap each call. Let's do the same.

  // Determine if the line is a "let X = EXPR" assignment.
  const isAssignment = /^(\s*)(let|let mut)\s+(\([^)]+\)|\w+)\s*=/.test(line);
  if (isAssignment) {
    const m = line.match(/^(\s*)(let|let mut)\s+(\([^)]+\)|\w+)\s*=(.*?)(;)?\s*$/);
    if (!m) continue;
    const [, indent, letWord, lhs, rhs, semi] = m;
    // Replace all `&mut rng` with `r` in RHS.
    const newRhs = rhs.replace(/&mut rng/g, "r");
    // For multi-line RHS (where semi is undefined because the line ends with comma),
    // this approach won't work. We assume the line ends with `;` or `,`.
    const endsWithComma = newRhs.trimEnd().endsWith(",");
    if (semi) {
      // Single-line let binding. Wrap RHS in block.
      lines[i] = `${indent}${letWord} ${lhs} = { let r = &mut rng; ${newRhs.trimEnd()} };`;
      fixCount++;
    } else if (endsWithComma) {
      // Multi-line let binding. We can't wrap here without closing the block
      // properly. Skip and handle separately later.
    } else {
      // Unknown shape — skip.
    }
    continue;
  }

  // For non-assignment lines (assert!, function calls, etc.), wrap inline.
  // Note: this won't be syntactically correct for all lines; we only do it
  // when the line is a single-statement that can be a block expression.
  // For safety, only wrap lines whose trim starts with `assert!` or a known
  // call-site keyword.
  // Actually, simpler: leave non-assignment lines alone for now; they're
  // less common.
}

// For multi-line let bindings: find them and rewrite as shadow blocks.
// A multi-line let binding is a `let X = ...` followed by continuation lines
// (indented more deeply, ending with `;`). We want to wrap the WHOLE RHS.

let i = testStart;
while (i < lines.length) {
  const line = lines[i];
  if (/^\s*(let|let mut)\s+\w+\s*=/.test(line) && !line.includes("&mut rng")) {
    // Pure let with no &mut rng — skip.
    i++;
    continue;
  }
  if (/^\s*(let|let mut)\s+\w+\s*=.*[^;,]\s*$/.test(line) && !line.trimEnd().endsWith(";")) {
    // Multi-line let binding starting here.
    // Find the closing `;` (could span many lines).
    const startIndent = line.match(/^\s*/)[0];
    let j = i + 1;
    while (j < lines.length && !lines[j].trimEnd().endsWith(";")) j++;
    if (j >= lines.length) {
      i++;
      continue;
    }
    // We have lines [i..j] (inclusive). The first line has `let X = ...`.
    // The last line ends with `;`.
    const firstLine = lines[i];
    const lastLine = lines[j];
    // Find the RHS: everything after the first `=` on line i.
    const eqIdx = firstLine.indexOf("=");
    const lhs = firstLine.substring(0, eqIdx + 1);
    let rhs = firstLine.substring(eqIdx + 1);
    // Build the RHS body: lines [i+1..j-1] + the prefix of lastLine before `;`.
    const bodyLines = [];
    bodyLines.push(rhs.trimEnd().endsWith(",") ? rhs : rhs.replace(/,\s*$/, ""));
    for (let k = i + 1; k < j; k++) bodyLines.push(lines[k]);
    // Last line: everything before the `;`.
    bodyLines.push(lastLine.replace(/;\s*$/, ""));
    // Now wrap: rewrite first line as `let X = { let r = &mut rng; <rewritten first rhs> };`
    // and inner lines: just replace `&mut rng` with `r`.
    // The last line: replace `&mut rng` with `r` and append `};`.
    const firstRhsNew = bodyLines[0].replace(/&mut rng/g, "r");
    lines[i] = `${lhs} { let r = &mut rng; ${firstRhsNew}`;
    for (let k = i + 1; k < j; k++) {
      lines[k] = lines[k].replace(/&mut rng/g, "r");
    }
    lines[j] = lines[j].replace(/&mut rng/g, "r").replace(/;\s*$/, " };");
    fixCount++;
    i = j + 1;
  } else {
    i++;
  }
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Wrapped ${fixCount} lines with shadow r.`);