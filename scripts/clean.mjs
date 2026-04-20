import { rmSync } from 'node:fs';
import { join, resolve } from 'node:path';

const targetRoot = resolve(process.argv[2] || process.cwd());
const targets = [
  join(targetRoot, 'dist'),
  join(targetRoot, 'tsconfig.tsbuildinfo'),
  join(targetRoot, '.turbo'),
];

for (const target of targets) {
  rmSync(target, { recursive: true, force: true });
}
