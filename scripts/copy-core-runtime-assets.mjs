import { copyFile, mkdir, cp, readFile, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const coreWorkerSource = resolve(rootDir, "packages/core/src/inference/worker.js");
const coreWorkerDist = resolve(rootDir, "packages/core/dist/inference/worker.js");
const appCfpSource = resolve(rootDir, "packages/core/cfp.py");
const appCfpDist = resolve(rootDir, "app/dist/packages/core/cfp.py");
const appWorkerDist = resolve(rootDir, "app/dist/packages/core/dist/inference/worker.js");
const appIndexSource = resolve(rootDir, "app/index.html");
const appIndexDist = resolve(rootDir, "app/dist/index.html");
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
// copy app static files (index.html, service worker)
try {
  // read and rewrite index.html so importmap paths are relative to app/dist
  const raw = await readFile(appIndexSource, "utf8");
  let fixed = raw.replace(/"\.\./g, '".');
  // when serving from app/dist, the bundle is at ./index.js not ./dist/index.js
  fixed = fixed.replace('src="./dist/index.js"', 'src="./index.js"');
  await mkdir(dirname(appIndexDist), { recursive: true });
  await writeFile(appIndexDist, fixed, "utf8");
} catch (e) {
  // ignore if not present
}
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
  const modelsDest = resolve(rootDir, `app/dist/packages/core/models`);
  await mkdir(modelsDest, { recursive: true });
  await cp(modelsSrc, modelsDest, { recursive: true });
} catch (e) {
  // ignore if not present
}
