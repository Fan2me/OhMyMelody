import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { watch } from "node:fs";
import { access, readFile, stat } from "node:fs/promises";
import { dirname, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const rootDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const port = Number(process.env.PORT || 4173);
const pnpmCommand = "pnpm";
const watchedPaths = [
  ["package.json", false],
  ["pnpm-workspace.yaml", false],
  ["turbo.json", false],
  ["app/index.html", false],
  ["app/package.json", false],
  ["app/tsconfig.json", false],
  ["app/src", true],
  ["packages/core/package.json", false],
  ["packages/core/tsconfig.json", false],
  ["packages/core/src", true],
  ["packages/runtime/package.json", false],
  ["packages/runtime/tsconfig.json", false],
  ["packages/runtime/src", true],
  ["packages/ui/package.json", false],
  ["packages/ui/tsconfig.json", false],
  ["packages/ui/src", true],
];

let buildInFlight = false;
let buildAgain = false;
let shuttingDown = false;
const watchers = new Set();
const rebuildTimers = new Map();

function log(message) {
  console.log(`[dev] ${message}`);
}

function runBuild() {
  return new Promise((resolve, reject) => {
    const child = spawn(pnpmCommand, ["build"], {
      cwd: rootDir,
      stdio: "inherit",
      shell: process.platform === "win32",
    });

    child.on("error", reject);
    child.on("exit", (code, signal) => {
      if (code === 0) {
        resolve();
        return;
      }
      reject(new Error(signal ? `build exited with signal ${signal}` : `build exited with code ${code ?? "unknown"}`));
    });
  });
}

async function rebuild() {
  if (shuttingDown) {
    return;
  }
  if (buildInFlight) {
    buildAgain = true;
    return;
  }

  buildInFlight = true;
  try {
    log("building workspace...");
    await runBuild();
    log("build complete");
  } catch (error) {
    console.error("[dev] build failed:", error);
  } finally {
    buildInFlight = false;
    if (buildAgain) {
      buildAgain = false;
      void rebuild();
    }
  }
}

function scheduleRebuild(pathname) {
  const previous = rebuildTimers.get(pathname);
  if (previous) {
    clearTimeout(previous);
  }
  const timer = setTimeout(() => {
    rebuildTimers.delete(pathname);
    void rebuild();
  }, 120);
  rebuildTimers.set(pathname, timer);
}

function startWatcher(target, recursive) {
  try {
    const watcher = watch(
      resolve(rootDir, target),
      { recursive },
      () => scheduleRebuild(target),
    );
    watchers.add(watcher);
  } catch (error) {
    console.warn(`[dev] watcher unavailable for ${target}:`, error instanceof Error ? error.message : error);
  }
}

function contentTypeFor(filePath) {
  switch (extname(filePath)) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
    case ".mjs":
      return "application/javascript; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    case ".svg":
      return "image/svg+xml";
    case ".png":
      return "image/png";
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".woff":
      return "font/woff";
    case ".woff2":
      return "font/woff2";
    default:
      return "application/octet-stream";
  }
}

async function sendFile(response, filePath) {
  try {
    const fileStat = await stat(filePath);
    if (fileStat.isDirectory()) {
      await sendFile(response, resolve(filePath, "index.html"));
      return;
    }
    const body = await readFile(filePath);
    response.writeHead(200, {
      "content-type": contentTypeFor(filePath),
      "cache-control": "no-store",
    });
    response.end(body);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
}

const server = createServer(async (request, response) => {
  const requestUrl = request.url ? new URL(request.url, `http://127.0.0.1:${port}`) : new URL("/", `http://127.0.0.1:${port}`);
  let pathname = decodeURIComponent(requestUrl.pathname);
  if (pathname === "/") {
    response.writeHead(302, { location: "/app/index.html" });
    response.end();
    return;
  }
  if (pathname === "/app") {
    response.writeHead(302, { location: "/app/index.html" });
    response.end();
    return;
  }

  if (pathname.endsWith("/")) {
    pathname += "index.html";
  }

  const resolvedPath = resolve(rootDir, `.${pathname}`);
  const rootWithSep = rootDir.endsWith(sep) ? rootDir : `${rootDir}${sep}`;
  if (resolvedPath !== rootDir && !resolvedPath.startsWith(rootWithSep)) {
    response.writeHead(403, { "content-type": "text/plain; charset=utf-8" });
    response.end("Forbidden");
    return;
  }

  await sendFile(response, resolvedPath);
});

async function main() {
  for (const [target, recursive] of watchedPaths) {
    startWatcher(target, recursive);
  }

  await rebuild();
  await listenWithFallback(port);
}

async function listenWithFallback(startPort) {
  const maxAttempts = 10;
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    const nextPort = startPort + attempt;
    try {
      await new Promise((resolve, reject) => {
        const onError = (error) => {
          server.off("listening", onListening);
          reject(error);
        };
        const onListening = () => {
          server.off("error", onError);
          resolve();
        };
        server.once("error", onError);
        server.once("listening", onListening);
        server.listen(nextPort, "127.0.0.1");
      });
      log(`server running at http://127.0.0.1:${nextPort}/app/index.html`);
      return;
    } catch (error) {
      const code = error && typeof error === "object" ? error.code : null;
      if (code !== "EADDRINUSE" || attempt === maxAttempts - 1) {
        throw error;
      }
      log(`port ${nextPort} is busy, trying ${nextPort + 1}`);
    }
  }
  throw new Error("unable to start dev server");
}

function shutdown(signal) {
  if (shuttingDown) {
    return;
  }
  shuttingDown = true;
  log(`shutting down (${signal})`);
  for (const timer of rebuildTimers.values()) {
    clearTimeout(timer);
  }
  rebuildTimers.clear();
  for (const watcher of watchers) {
    try {
      watcher.close();
    } catch {}
  }
  watchers.clear();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 1000).unref?.();
}

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

main().catch((error) => {
  console.error("[dev] failed to start:", error);
  process.exit(1);
});
