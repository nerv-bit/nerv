// Delete the two orphan test modules from bin/nerv-node/src/wiring.rs:
//   - `gossip_engine_tests` (lines 1293..=1844): uses crate::host / crate::testutil
//     that no longer exist in the bin crate.
//   - `gap7_knowledge_tests` (lines 1847..=2076): uses Host/HostConfig without
//     the corresponding `use` import.
// Keep the first `mod tests` (1014..=1290), which is self-contained.

const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/bin/nerv-node/src/wiring.rs";

let src = fs.readFileSync(path, "utf8");
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");
const total = lines.length;

// Sanity checks.
if (!lines[1013].trim().startsWith("mod tests")) {
  console.error("Line 1014 expected 'mod tests':", lines[1013]);
  process.exit(2);
}
if (!lines[1294].trim().startsWith("mod gossip_engine_tests")) {
  console.error("Line 1295 expected 'mod gossip_engine_tests':", lines[1294]);
  process.exit(2);
}
if (!lines[1848].trim().startsWith("mod gap7_knowledge_tests")) {
  console.error("Line 1849 expected 'mod gap7_knowledge_tests':", lines[1848]);
  process.exit(2);
}

// Keep: 1..1290 (mod tests ends here)
// Skip: 1293..2076 (gossip_engine_tests + gap7_knowledge_tests, plus their
//       surrounding `#[cfg(test)]` attributes and blank-line separators)
// Keep: 2077-end (file trailing newline)
const head = lines.slice(0, 1290); // 0-indexed: up to index 1289, i.e. lines 1..1290
const tail = lines.slice(2076);    // lines 2077-end
const out = head.concat(tail);

fs.writeFileSync(path, out.join("\r\n"));
console.log(`Trimmed: ${lines.length} -> ${out.length} lines.`);