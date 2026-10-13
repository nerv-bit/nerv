// More aggressive: handle `leg(&mut rng, ..., h(&mut rng))` patterns where
// there's no shadow block. Extract h(&mut rng) to a let binding before.

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

// Find lines that have BOTH `leg(&mut rng` AND `h(&mut rng)` in the same line.
// Skip lines that already have `__anchorN` (already extracted) or are inside
// shadow blocks (have `let r = &mut rng`).
let insertions = [];
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  if (!/leg\s*\(\s*&mut rng/.test(line)) continue;
  const hMatches = [...line.matchAll(/\bh\s*\(\s*&mut rng\s*\)/g)];
  if (hMatches.length === 0) continue;
  if (line.match(/__anchor\d+/)) continue;
  // Skip lines inside shadow blocks: they have `let r = &mut rng` already.
  if (line.includes("let r = &mut rng")) continue;
  insertions.push({ lineIdx: i, hMatches });
}

let anchorCounter = 100; // start at 100 to avoid clash with previous
for (let k = insertions.length - 1; k >= 0; k--) {
  const { lineIdx, hMatches } = insertions[k];
  let line = lines[lineIdx];
  const indent = line.match(/^\s*/)[0];

  // Replace each h(&mut rng) with __anchorN.
  for (let j = hMatches.length - 1; j >= 0; j--) {
    const m = hMatches[j];
    anchorCounter++;
    const anchorName = `__h${anchorCounter}`;
    line = line.substring(0, m.index) + anchorName + line.substring(m.index + m[0].length);
  }

  const prelude = [];
  for (let j = 1; j <= hMatches.length; j++) {
    const anchorName = `__h${anchorCounter - hMatches.length + j}`;
    prelude.push(`${indent}let ${anchorName} = h(&mut rng);`);
  }
  lines[lineIdx] = line;
  lines.splice(lineIdx, 0, ...prelude);
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Extracted ${anchorCounter - 100} h(&mut rng) calls.`);