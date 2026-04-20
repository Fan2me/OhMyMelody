import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";
import test from "node:test";

function findEdgeExecutable() {
  const candidates = [
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  ];
  return candidates.find((path) => existsSync(path)) || null;
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
    case ".wasm":
      return "application/wasm";
    case ".py":
      return "text/x-python; charset=utf-8";
    default:
      return "application/octet-stream";
  }
}

function createStaticServer(rootDir) {
  const server = createServer(async (request, response) => {
    const requestUrl = request.url
      ? new URL(request.url, "http://127.0.0.1")
      : new URL("/", "http://127.0.0.1");
    let pathname = decodeURIComponent(requestUrl.pathname);

    if (pathname === "/__worker_test__.html") {
      response.writeHead(200, {
        "content-type": "text/html; charset=utf-8",
        "cache-control": "no-store",
      });
      response.end("<!doctype html><html><head><meta charset=\"utf-8\"><title>worker smoke</title></head><body></body></html>");
      return;
    }

    if (pathname === "/") {
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

    try {
      const body = await readFile(resolvedPath);
      response.writeHead(200, {
        "content-type": contentTypeFor(resolvedPath),
        "cache-control": "no-store",
      });
      response.end(body);
    } catch {
      response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
      response.end("Not found");
    }
  });

  return server;
}

async function listen(server) {
  await new Promise((resolveListen, rejectListen) => {
    server.once("error", rejectListen);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", rejectListen);
      resolveListen();
    });
  });
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("failed to bind test server");
  }
  return address.port;
}

async function waitForWorkerInit(page) {
  return await page.evaluate(async () => {
    const worker = new Worker("/packages/core/dist/cfp/worker.js", {
      type: "module",
    });
    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error("worker init timed out"));
      }, 300000);

      worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.cmd === "inited") {
          clearTimeout(timeout);
          worker.terminate();
          resolve("inited");
          return;
        }
        if (message.cmd === "error") {
          clearTimeout(timeout);
          worker.terminate();
          reject(new Error(message.error || "worker error"));
        }
      };
      worker.onerror = () => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error("worker runtime error"));
      };

      worker.postMessage({
        cmd: "init",
        cfpScriptUrl: "/packages/core/cfp.py",
        pyodideScriptUrl:
          "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/pyodide.js",
        pyodideIndexURL:
          "https://cdn.jsdelivr.net/pyodide/v0.25.1/full/",
      });
    });
    return result;
  });
}

async function waitForInferenceWorkerInit(page) {
  return await page.evaluate(async () => {
    const worker = new Worker("/packages/core/dist/inference/worker.js");
    const result = await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error("inference worker init timed out"));
      }, 300000);

      worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.cmd === "inited") {
          clearTimeout(timeout);
          worker.terminate();
          resolve("inited");
          return;
        }
        if (message.cmd === "error") {
          clearTimeout(timeout);
          worker.terminate();
          reject(new Error(message.error || "inference worker error"));
        }
      };
      worker.onerror = () => {
        clearTimeout(timeout);
        worker.terminate();
        reject(new Error("inference worker runtime error"));
      };

      worker.postMessage({
        cmd: "init",
        modelName: "msnet_80.98.onnx",
      });
    });
    return result;
  });
}

test("browser worker initializes on localhost", async () => {
  const edgeExecutable = findEdgeExecutable();
  assert.ok(edgeExecutable, "Microsoft Edge executable was not found");

  const rootDir = resolve(fileURLToPath(new URL("../../", import.meta.url)));
  const server = createStaticServer(rootDir);
  const port = await listen(server);
  const browser = await chromium.launch({
    executablePath: edgeExecutable,
    headless: true,
  });

  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/__worker_test__.html`, {
      waitUntil: "domcontentloaded",
    });

    const initResult = await waitForWorkerInit(page);
    assert.equal(initResult, "inited");

    const inferenceInitResult = await waitForInferenceWorkerInit(page);
    assert.equal(inferenceInitResult, "inited");
  } finally {
    await browser.close();
    await new Promise((resolveClose) => server.close(() => resolveClose()));
  }
});
