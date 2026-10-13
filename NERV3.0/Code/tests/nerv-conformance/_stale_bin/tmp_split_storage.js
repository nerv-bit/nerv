// Split crates/nerv-wallet-core/tests/invariants.rs into two files.
// Lines 1-237 stay in invariants.rs (ending at the last test's `}`).
// Lines 241-end (skipping the `File:` marker at 239 and blank line at 240)
// become the new crates/nerv-wallet-core/tests/storage.rs.

const fs = require("fs");
const invPath = "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-wallet-core/tests/invariants.rs";
const stoPath = "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-wallet-core/tests/storage.rs";

let src = fs.readFileSync(invPath, "utf8");
// Normalize CRLF -> LF for indexing.
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");

// Sanity check.
if (!lines[236].trim().startsWith("}")) {
  console.error(`Line 237 expected to start with }: "${lines[236]}"`);
  process.exit(2);
}
if (!lines[238].trim().startsWith("File: ")) {
  console.error(`Line 239 expected to be a File: marker: "${lines[238]}"`);
  process.exit(2);
}

const invLines = lines.slice(0, 237);
const stoLines = lines.slice(240); // drop the File: marker and the blank line after it

fs.writeFileSync(invPath, invLines.join("\r\n"));
fs.writeFileSync(stoPath, stoLines.join("\r\n"));
console.log(`invariants.rs -> ${invLines.length} lines; storage.rs -> ${stoLines.length} lines`);