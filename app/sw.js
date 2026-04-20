const CACHE_NAME = "ohm-app-v2";
const CACHE_PREFIX = "ohm-app-";
const CDN_HOSTS = new Set(["cdn.jsdelivr.net"]);
const PYODIDE_CDN_VERSION = "0.25.1";
const ORT_CDN_VERSION = "1.24.3";
const APP_BASE_URL = new URL(".", self.location.href);
const CORE_RUNTIME_PREFIX = new URL("./packages/core/dist/", APP_BASE_URL).pathname;
const CORE_CFP_SCRIPT_PATH = new URL("./packages/core/cfp.py", APP_BASE_URL).pathname;
const DEFAULT_WARM_CORE_URLS = [
  new URL("./packages/core/dist/cfp/worker.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/cfp/chunk.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/cfp/common.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/cfp/cfp.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/cfp/pyodide-bootstrap.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/script/script.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/inference/inference.js", APP_BASE_URL).toString(),
  new URL("./packages/core/dist/inference/worker.js", APP_BASE_URL).toString(),
];
const DEFAULT_WARM_VENDOR_URLS = [
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.js`,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide-lock.json`,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.asm.js`,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/pyodide.asm.wasm`,
  `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_CDN_VERSION}/full/python_stdlib.zip`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort.min.js`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort.webgpu.min.js`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort-wasm-simd-threaded.asyncify.mjs`,
  `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_CDN_VERSION}/dist/ort-wasm-simd-threaded.asyncify.wasm`,
];

function isCacheableRequest(request) {
  if (request.method !== "GET") {
    return false;
  }
  const url = new URL(request.url);
  return url.origin === self.location.origin
    ? url.pathname === CORE_CFP_SCRIPT_PATH || url.pathname.startsWith(CORE_RUNTIME_PREFIX)
    : CDN_HOSTS.has(url.hostname);
}

async function cacheFirst(request) {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  if (response && response.ok) {
    cache.put(request, response.clone()).catch(() => {});
  }
  return response;
}

async function warmAssets(urls) {
  const cache = await caches.open(CACHE_NAME);
  await Promise.all(
    urls.map(async (url) => {
      try {
        await cache.add(url);
      } catch {}
    }),
  );
}

async function warmCoreCfpAsset() {
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.add(new URL("./packages/core/cfp.py", APP_BASE_URL).toString());
  } catch {}
}

async function warmCoreRuntimeAssets() {
  await warmAssets(DEFAULT_WARM_CORE_URLS);
  await warmCoreCfpAsset();
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    await self.skipWaiting();
    await warmAssets(DEFAULT_WARM_VENDOR_URLS);
    await warmCoreRuntimeAssets();
  })());
});

self.addEventListener("activate", (event) => {
  event.waitUntil((async () => {
    const cacheNames = await caches.keys();
    await Promise.all(
      cacheNames.map(async (name) => {
        if (name.startsWith(CACHE_PREFIX) && name !== CACHE_NAME) {
          await caches.delete(name);
        }
      }),
    );
    await self.clients.claim();
  })());
});

self.addEventListener("message", (event) => {
  const data = event.data || {};
  if (data.type === "warm-default-assets") {
    event.waitUntil((async () => {
      await warmAssets(DEFAULT_WARM_VENDOR_URLS);
      await warmCoreRuntimeAssets();
    })());
  }
  if (data.type === "warm-assets" && Array.isArray(data.urls)) {
    event.waitUntil(warmAssets(data.urls));
  }
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (!isCacheableRequest(request)) {
    return;
  }
  event.respondWith(cacheFirst(request));
});
