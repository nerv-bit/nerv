// For lines of the form:
//   let X = { let r = &mut rng; CALL(r, ..., vec![{ let __nf = h(r); __nf }], ...) };
// rewrite to:
//   let __nf = h(&mut rng);
//   let X = { let r = &mut rng; CALL(r, ..., vec![__nf], ...) };
// i.e., extract the nullifier out so we don't re-borrow r inside the vec literal.
//
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

// For each line that has the shadow r + h(r) inside vec! literal:
// We need to insert a `let __nf = h(&mut rng);` line BEFORE this line.
let insertions = [];
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  // Match the pattern: vec![{ let __nf = h(r); __nf }]
  if (line.includes("vec![{ let __nf = h(r); __nf }]")) {
    insertions.push(i);
  }
}

// Insert in reverse order so indices don't shift.
for (let k = insertions.length - 1; k >= 0; k--) {
  const i = insertions[k];
  const line = lines[i];
  // Determine indent.
  const indent = line.match(/^\s*/)[0];
  // Build the new line: replace `vec![{ let __nf = h(r); __nf }]` with `vec![__nf]`.
  const newLine = line.replace(
    "vec![{ let __nf = h(r); __nf }]",
    "vec![__nf_h]"
  );
  lines[i] = newLine;
  // Insert `let __nf_h = h(&mut rng);` BEFORE this line.
  lines.splice(i, 0, `${indent}let __nf_h = h(&mut rng);`);
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Extracted ${insertions.length} nullifier expressions.`);