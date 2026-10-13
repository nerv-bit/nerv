// Wrap lines in `crates/nerv-state/src/executor.rs` that contain TWO or more
// `&mut rng` references with a `{ let r = &mut rng; ... }` block, rewriting the
// subsequent occurrences to `r`. This is the proven fix for the
// "cannot borrow `rng` as mutable more than once at a time" errors when one
// expression borrows rng twice.
//
// Only operates on lines inside `mod tests { ... }` to avoid touching the
// production code.

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

// Find lines that contain >= 2 occurrences of `&mut rng`.
let fixCount = 0;
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  // Skip already-wrapped shadow blocks (heuristic: line starts with `let r = ...`).
  // We also skip lines that are short comments or have only one &mut rng.
  const occurrences = (line.match(/&mut rng/g) || []).length;
  if (occurrences < 2) continue;

  // Skip if this line is part of an existing shadow block we already wrote.
  // (Already-wrapped lines contain `let r = &mut rng;` and use `r` not `rng`.)
  if (line.includes("let r = &mut rng") || line.includes("let __nf")) {
    continue;
  }

  // Also skip lines that don't start with a typical call-site keyword
  // (`let`, `assert!`, etc.) — those are unlikely to be problematic.
  const trimmed = line.trimStart();
  if (!trimmed.startsWith("let ") && !trimmed.startsWith("assert!") &&
      !trimmed.startsWith("if ") && !trimmed.startsWith("apply_block") &&
      !trimmed.startsWith("for ") && !trimmed.startsWith("match ") &&
      !trimmed.startsWith("return ")) {
    continue;
  }

  // Rewrite: introduce `let r = &mut rng; `, replace subsequent `&mut rng` with `r`.
  // Determine indent from the original line.
  const indent = line.match(/^\s*/)[0];
  // First &mut rng becomes `r` (well, we use `let r = &mut rng`).
  // Actually, the standard fix is to wrap with `{ let r = &mut rng; ... }`.
  // The first &mut rng stays as `r` (used as the argument to the outer call).
  // So we rewrite the line to:
  //   { let r = &mut rng; <original line with each subsequent &mut rng replaced by r> }
  // First &mut rng is the one inside `let r = &mut rng`.
  let rewritten = line.replace(/&mut rng/g, (match, offset, fullLine) => {
    if (offset === line.indexOf("&mut rng")) {
      return "r"; // first occurrence becomes the borrowed `r`
    }
    return "r";
  });

  // Now we need to inject `let r = &mut rng;` after the `{`. The simplest way:
  // wrap with `{ let r = &mut rng; <content> }`. But we need to be careful that
  // the assignment target (e.g. `let x = `) stays OUTSIDE the block.
  // The pattern is `let x = EXPR;` — we want to make it `let x = { let r = &mut rng; EXPR_WITH_R };`.
  // For assert! / apply_block / etc., we just wrap.
  const isAssignment = /^\s*(let|let mut)\s+\w+\s*=\s*/.test(line);
  if (isAssignment) {
    // Find the `=` sign position and the body.
    const eqMatch = line.indexOf("=");
    // Split into LHS and RHS.
    const lhs = line.substring(0, eqMatch + 1);
    let rhs = line.substring(eqMatch + 1);
    // Strip trailing semicolon from RHS if present.
    rhs = rhs.replace(/;\s*$/, "");
    // Replace all `&mut rng` in rhs with `r`.
    rhs = rhs.replace(/&mut rng/g, "r");
    // Build new line: `let x = { let r = &mut rng; <rhs> };`
    rewritten = `${lhs} { let r = &mut rng; ${rhs} };`;
  } else {
    // Wrap with shadow block.
    rewritten = `{ let r = &mut rng; ${line.trimEnd()} }`;
  }

  lines[i] = rewritten;
  fixCount++;
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Wrapped ${fixCount} lines with shadow r.`);