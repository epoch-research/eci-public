// Authoritative subset-ECI fit: imports the REAL website fitModelECI from
// epoch-website-astro/legacy/vizs/benchmarks/eciSubsetMath.ts (unmodified) and
// runs it per model. Input/output are JSON files (built/consumed by fit_subset.py).
//   node --import ./scripts/_website_fit_register.mjs ./scripts/website_fit.mjs in.json out.json
// Override the .ts path with WEBSITE_ECI_TS.
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const tsPath = process.env.WEBSITE_ECI_TS
  || path.resolve(here, '../../epoch-website-astro/legacy/vizs/benchmarks/eciSubsetMath.ts');
const { fitModelECI } = await import(pathToFileURL(tsPath).href);

const [inPath, outPath] = process.argv.slice(2);
const input = JSON.parse(readFileSync(inPath, 'utf8'));
const out = {};
for (const item of input.items)
  out[item.id] = fitModelECI(item.obs.map(([perf, edi, slope]) => ({ perf, edi, slope })));
writeFileSync(outPath, JSON.stringify(out));
console.error(`website_fit: fitted ${input.items.length} item(s) using ${tsPath}`);
