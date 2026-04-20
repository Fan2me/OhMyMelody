import { mkdir, copyFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourcePath = resolve(rootDir, "packages/core/cfp.py");
const destinationPath = resolve(rootDir, "app/dist/packages/core/cfp.py");

await mkdir(dirname(destinationPath), { recursive: true });
await copyFile(sourcePath, destinationPath);
