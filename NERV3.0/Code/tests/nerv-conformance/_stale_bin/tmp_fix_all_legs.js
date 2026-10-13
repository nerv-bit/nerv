// Fix ALL `leg(&mut rng, ..., h(&mut rng))` patterns by extracting h(&mut rng)
// to a let binding BEFORE the leg call. Works for both one-line and multi-line
// leg(...) call sites.

const fs = require("fs");
const files = [
  "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/executor.rs",
  "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/fraud.rs",
  "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/testkit.rs",
];

for (const path of files) {
  let src = fs.readFileSync(path, "utf8");
  src = src.replace(/\r\n/g, "\n");
  const lines = src.split("\n");

  // Find each multi-line leg(...) call that contains a `h(&mut rng)` somewhere.
  // A leg call starts with `leg(` and ends with `)` on a line whose opening
  // parens balance.
  //
  // Simpler approach: for each line containing both `leg(&mut rng` AND
  // `h(&mut rng)`, extract the h(&mut rng) to a let binding before this line,
  // and replace h(&mut rng) with __hN in the line.

  let counter = 200;
  const newLines = [];
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Skip if this is inside a shadow block (line already has `let r = &mut rng`).
    if (line.includes("let r = &mut rng")) {
      newLines.push(line);
      continue;
    }
    // Skip lines that already have __hN.
    if (/__h\d+/.test(line)) {
      newLines.push(line);
      continue;
    }
    // Find h(&mut rng) in this line.
    const m = line.match(/\bh\s*\(\s*&mut rng\s*\)/);
    if (!m) {
      newLines.push(line);
      continue;
    }
    // Confirm line also has leg(&mut rng.
    if (!/leg\s*\(\s*&mut rng/.test(line)) {
      newLines.push(line);
      continue;
    }
    const indent = line.match(/^\s*/)[0];
    counter++;
    const newLine = line.replace(m[0], `__h${counter}`);
    newLines.push(`${indent}let __h${counter} = h(&mut rng);`);
    newLines.push(newLine);
  }

  fs.writeFileSync(path, newLines.join("\r\n"));
  console.log(`${path.split('/').pop()}: extracted ${counter - 200} more h(&mut rng) calls.`);
}