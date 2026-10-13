// Fix lines like:
//   let X = { let r = &mut rng; call(...); };
// to:
//   let X = { let r = &mut rng; call(...) };
// by removing the inner trailing semicolon before `}`.
// Only operate inside `mod tests`.

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

let count = 0;
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  // Match the pattern: `...); };` (closing the inner block + outer)
  // We want to change `...); };` to `...) };`
  if (/\); \};$/.test(line.trimEnd())) {
    lines[i] = line.replace(/\); \};$/, ") };");
    count++;
  }
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Removed inner semicolons in ${count} lines.`);