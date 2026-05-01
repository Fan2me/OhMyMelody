import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

const args = process.argv.slice(2);
const packageRoot = resolve(args[0] || process.cwd());
const packageArgs = args.slice(1);

const result = spawnSync(
  'oxlint',
  [packageRoot, ...packageArgs],
  {
    stdio: 'inherit',
    shell: true,
  },
);

if (result.error) {
  throw result.error;
}

process.exit(result.status ?? 1);
