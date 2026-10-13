// Mechanical fixes for bin/nerv-node/src/producer.rs:
//   - All `ShardId::new(a, b)` arguments need `.unwrap()` since the new ShardId
//     constructor returns Result<ShardId, TypeError>.
//   - Replace `SigningKey::generate()` with the available alternative:
//     `SigningKey::from_seed(&[0xAB; 32])` (or similar) — the producer.rs tests
//     only need any working signing key.
//
// We restrict edits to lines inside the `mod tests` block (the test scaffolding
// issue) so production code is untouched.

const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/bin/nerv-node/src/producer.rs";

let src = fs.readFileSync(path, "utf8");
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");

// Find the `mod tests {` start.
let testStart = -1;
for (let i = 0; i < lines.length; i++) {
  if (lines[i].trim() === "mod tests {") {
    testStart = i;
    break;
  }
}
if (testStart < 0) {
  console.error("Could not find `mod tests {` line.");
  process.exit(2);
}
console.log(`Mod tests starts at line ${testStart + 1}.`);

// Mechanical substitution 1: replace `ShardId::new(a, b)` with `ShardId::new(a, b).unwrap()`
// ONLY inside the test block. Use a regex that captures arguments safely.
let inTest = false;
let braceDepth = 0;
let fixCount = 0;
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  if (!inTest) {
    // Count braces to find when we leave the mod tests.
    if (line.includes("{")) {
      braceDepth++;
      inTest = true;
    }
    continue;
  }
  // Now we're inside the test mod.
  // Match `ShardId::new(a, b)` where a, b are simple number literals.
  // Don't double-wrap if already followed by `.unwrap()`.
  let newLine = line;
  newLine = newLine.replace(
    /\bShardId::new\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)(?!\s*\.\s*unwrap)/g,
    (m, a, b) => {
      fixCount++;
      return `ShardId::new(${a}, ${b}).unwrap()`;
    }
  );
  lines[i] = newLine;
}

// Mechanical substitution 2: `SigningKey::generate()` -> `SigningKey::from_seed(&[0x42; 32])`
// Only in the test block.
let genFixed = 0;
for (let i = testStart; i < lines.length; i++) {
  const line = lines[i];
  if (line.includes("SigningKey::generate()")) {
    lines[i] = line.replace("SigningKey::generate()", "SigningKey::from_seed(&[0x42; 32])");
    genFixed++;
  }
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`ShardId::new() fixes: ${fixCount}; SigningKey::generate() fixes: ${genFixed}.`);