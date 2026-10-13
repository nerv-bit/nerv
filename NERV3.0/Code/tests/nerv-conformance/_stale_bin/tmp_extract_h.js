// Find lines with the pattern `leg(r, ..., h(r))` and extract the inner `h(r)`.
// Rewrite as:
//   let __anchor = h(&mut rng);
//   let X = leg(&mut rng, ..., __anchor);
// Operates inside `mod tests` only.

const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/executor.rs";

let src = fs.readFileSync(path, "utf8");
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");

let testStart = -1;
for (let i = 0; i < lines.length; i++) {
  if (lines[i].trim() === "mod tests {") {
    testStart = i;
    break;
  }
}

// Lines that match: ... leg(r, ..., h(r)) ...
// We want to extract `h(r)` and put it on a prior line.
// Pattern: the shadow-wrapped `leg(r, ..., h(r))` form.
let insertions = [];
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  // Match lines that have both `leg(r,` AND `h(r)` (or `h(r))`).
  if (!/leg\s*\(\s*r\s*,/.test(line)) continue;
  // Find occurrences of `h(r)` in this line.
  const hrMatches = [...line.matchAll(/\bh\s*\(\s*r\s*\)/g)];
  if (hrMatches.length === 0) continue;

  // For each match, we'll insert a let binding before this line.
  // To avoid double-extraction, only do lines that don't already have
  // their anchor extracted.
  if (line.match(/__anchor\d+/)) continue;

  insertions.push({ lineIdx: i, hrMatches });
}

// Insert in reverse order. We track a unique anchor index to avoid clashes.
let anchorCounter = 0;
for (let k = insertions.length - 1; k >= 0; k--) {
  const { lineIdx, hrMatches } = insertions[k];
  let line = lines[lineIdx];
  const indent = line.match(/^\s*/)[0];

  // Replace each `h(r)` with `__anchor_N` (in reverse order so indices stay valid).
  for (let j = hrMatches.length - 1; j >= 0; j--) {
    const m = hrMatches[j];
    anchorCounter++;
    const anchorName = `__anchor${anchorCounter}`;
    line = line.substring(0, m.index) + anchorName + line.substring(m.index + m[0].length);
  }

  // Build the pre-extraction lines: one `let __anchorN = h(&mut rng);` per anchor.
  const prelude = [];
  for (let j = 1; j <= hrMatches.length; j++) {
    const anchorName = `__anchor${anchorCounter - hrMatches.length + j}`;
    prelude.push(`${indent}let ${anchorName} = h(&mut rng);`);
  }
  // Replace `leg(r,` with `leg(&mut rng,` (since `r` is shadowed inside the
  // shadow block, but the anchor vars are outside; the closure call gets
  // the shadow). Actually since the shadow block has `let r = &mut rng;`
  // and we pass `&mut rng` instead of `r` here, the borrow checker treats
  // them as the same borrow.
  // But also: the surrounding block has `let r = &mut rng;` already, so
  // re-shadowing is fine. Just rewrite `leg(r,` to `leg(&mut rng,` to
  // avoid the second mutable borrow inside `leg`.
  // Actually we should keep `leg(r,` since we're inside the shadow block.
  // Wait — the shadow block has `let r = &mut rng;` and inside we have
  // `leg(r, ..., h(r))`. The `h(r)` inside `leg(r, ...)` is a re-borrow
  // which conflicts. So extract h(r) to a let binding OUTSIDE the shadow
  // block, then use the binding inside. The binding uses `&mut rng`
  // outside the shadow, which is fine.
  lines[lineIdx] = line;
  // Insert preludes before this line.
  lines.splice(lineIdx, 0, ...prelude);
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Extracted ${anchorCounter} h(r) calls.`);