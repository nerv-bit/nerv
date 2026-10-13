// Targeted UTF-8 line-based fixes for executor.rs corruption.
// Pattern: lines ending in just `;` (matching `^\s+;$`) need to become `};`.
// Plus: `TransactionShell { legs;` -> `TransactionShell { legs };`
// plus: `siblings: vec![],` followed by `           ;` (11-space indent) at line 1153.
const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/crates/nerv-state/src/executor.rs";
let src = fs.readFileSync(path, "utf8");
// Normalize CRLF -> LF for stable line indexing.
src = src.replace(/\r\n/g, "\n");
const lines = src.split("\n");

// Lines we identified as missing closing `};` (1-indexed).
// Conversion: append a `}` directly before the `;`.
const lineFixes = {
  268: "   };",   // close else block (3-space indent)
  280: "   };",
  360: "   };",   // close struct literal Applied {...}
  417: "   };",
  570: "                   };",  // 19-space indent
  777: "   };",
  796: "   };",
  804: "   };",
  834: "   };",
  850: "   };",   // close use import block
  1153: "           }];",  // close vec![] within struct literal SettledLeg {...} (11-space)
};

let changed = 0;
for (const [k, v] of Object.entries(lineFixes)) {
  const idx = parseInt(k, 10) - 1; // 0-indexed
  const cur = lines[idx];
  // current line should be `^\s+;$`
  if (!/^\s+;$/.test(cur)) {
    console.error(`Line ${k} unexpected: "${cur}"`);
    process.exit(2);
  }
  lines[idx] = v;
  changed++;
}

// Also fix line 1144: `TransactionShell { legs;` -> `TransactionShell { legs };`
{
  const idx = 1144 - 1;
  const cur = lines[idx];
  const want = cur.replace(/legs;$/, "legs };");
  if (cur === want || !/legs;$/.test(cur)) {
    console.error(`Line 1144 unexpected: "${cur}"`);
    process.exit(2);
  }
  lines[idx] = want;
  changed++;
}

fs.writeFileSync(path, lines.join("\r\n"));
console.log(`Applied ${changed} fixes.`);