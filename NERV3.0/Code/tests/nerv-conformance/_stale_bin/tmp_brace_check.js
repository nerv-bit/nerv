const fs = require("fs");
const path = "C:/Users/prosp/Documents/MiniMax NERV/tests/nerv-conformance/src/spec.rs";
const src = fs.readFileSync(path, "utf8");
let depth = 0;
let stack = [];
let i = 0;
let line = 1, col = 0;
while (i < src.length) {
  const c = src[i];
  if (c === "\n") { line++; col = 0; i++; continue; }
  col++;
  if (c === "/" && src[i+1] === "/") { while (i < src.length && src[i] !== "\n") i++; continue; }
  if (c === "/" && src[i+1] === "*") {
    i += 2;
    while (i < src.length && !(src[i] === "*" && src[i+1] === "/")) { if (src[i] === "\n") { line++; col = 0; } i++; }
    i += 2; continue;
  }
  if (c === "r" && src[i+1] === "#") {
    let h = "";
    let j = i + 2;
    while (j < src.length && src[j] === "#") { h += "#"; j++; }
    if (src[j] === '"') {
      j++;
      const close = '"' + h;
      while (j < src.length) {
        if (src[j] === "\n") { line++; col = 0; }
        if (src.substring(j, j + close.length) === close) { j += close.length; break; }
        j++;
      }
      i = j; continue;
    }
  }
  if (c === '"') { i++; while (i < src.length && src[i] !== '"') { if (src[i] === "\\") { i += 2; continue; } if (src[i] === "\n") { line++; col = 0; } i++; } i++; continue; }
  if (c === "'") { i++; while (i < src.length && src[i] !== "'") { if (src[i] === "\\") { i += 2; continue; } i++; } i++; continue; }
  if (c === "{") { depth++; stack.push({ line, col }); }
  else if (c === "}") { depth--; if (depth < 0) { console.log(`extra '}' at ${line}:${col}`); break; } else stack.pop(); }
  i++;
}
console.log("final depth =", depth);
console.log("unclosed '{' =", stack.length, stack[stack.length-1]);