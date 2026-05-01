import { copyFile, mkdir, cp } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const coreWorkerSource = resolve(rootDir, "packages/core/src/inference/worker.js");
const coreWorkerDist = resolve(rootDir, "packages/core/dist/inference/worker.js");
const appCfpSource = resolve(rootDir, "packages/core/cfp.py");
const appCfpDist = resolve(rootDir, "app/dist/packages/core/cfp.py");
const appWorkerDist = resolve(rootDir, "app/dist/packages/core/dist/inference/worker.js");
const appSwSource = resolve(rootDir, "app/sw.js");
const appSwDist = resolve(rootDir, "app/dist/sw.js");
const packagesToCopy = ["core", "runtime", "ui"];

async function copyFileSafe(sourcePath, destinationPath) {
  await mkdir(dirname(destinationPath), { recursive: true });
  await copyFile(sourcePath, destinationPath);
}

await copyFileSafe(coreWorkerSource, coreWorkerDist);
await copyFileSafe(coreWorkerSource, appWorkerDist);
await copyFileSafe(appCfpSource, appCfpDist);
try {
  await copyFileSafe(appSwSource, appSwDist);
} catch (e) {
  // ignore if not present
}

// copy packages/*/dist into app/dist/packages/* so importmap-relative paths work
for (const name of packagesToCopy) {
  const src = resolve(rootDir, `packages/${name}/dist`);
  const dest = resolve(rootDir, `app/dist/packages/${name}/dist`);
  try {
    await mkdir(dest, { recursive: true });
    await cp(src, dest, { recursive: true });
  } catch (e) {
    // skip missing packages
  }
}

// copy core models directory so resolveCoreModelUrl works
try {
  const modelsSrc = resolve(rootDir, `packages/core/models`);
  const coreDistModelsDest = resolve(rootDir, `packages/core/dist/models`);
  const appDistModelsDest = resolve(rootDir, `app/dist/packages/core/dist/models`);
  await mkdir(coreDistModelsDest, { recursive: true });
  await mkdir(appDistModelsDest, { recursive: true });
  await cp(modelsSrc, coreDistModelsDest, { recursive: true });
  await cp(modelsSrc, appDistModelsDest, { recursive: true });
} catch (e) {
  // ignore if not present
}
