const CACHE_NAME = "ohm-app-v5";
const CACHE_PREFIX = "ohm-app-";
const CDN_HOSTS = new Set(["cdn.jsdelivr.net"]);
const PYODIDE_CDN_VERSION = "0.25.1";
const ORT_CDN_VERSION = "1.24.3";
const APP_BASE_URL = new URL(".", self.location.href);
const APP_MANIFEST_URL = new URL("./manifest.json", APP_BASE_URL).toString();
const APP_SHELL_URLS = [
  new URL("./", APP_BASE_URL).toString(),
  new URL("./index.html", APP_BASE_URL).toString(),
  new URL("./app.js", APP_BASE_URL).toString(),
  new URL("./favicon.ico", APP_BASE_URL).toString(),
  new URL("./sw.js", APP_BASE_URL).toString(),
  APP_MANIFEST_URL,
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
  return url.origin === self.location.origin || CDN_HOSTS.has(url.hostname);
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

function collectManifestUrls(manifest) {
  const urls = new Set();
  if (!manifest || typeof manifest !== "object") {
    return urls;
  }
  for (const entry of Object.values(manifest)) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const asset = entry;
    if (typeof asset.file === "string") {
      urls.add(new URL(asset.file, APP_BASE_URL).toString());
    }
    if (Array.isArray(asset.css)) {
      for (const css of asset.css) {
        if (typeof css === "string") {
          urls.add(new URL(css, APP_BASE_URL).toString());
        }
      }
    }
    if (Array.isArray(asset.assets)) {
      for (const extra of asset.assets) {
        if (typeof extra === "string") {
          urls.add(new URL(extra, APP_BASE_URL).toString());
        }
      }
    }
  }
  return urls;
}

async function warmAppShellAssets() {
  await warmAssets(APP_SHELL_URLS);
}

async function warmBuildManifestAssets() {
  const cache = await caches.open(CACHE_NAME);
  try {
    const response = await fetch(APP_MANIFEST_URL, { cache: "no-store" });
    if (!response.ok) {
      return;
    }
    const manifest = await response.json();
    const urls = collectManifestUrls(manifest);
    await Promise.all(
      Array.from(urls).map(async (url) => {
        try {
          await cache.add(url);
        } catch {}
      }),
    );
  } catch {}
}

self.addEventListener("install", (event) => {
  event.waitUntil((async () => {
    await self.skipWaiting();
    await warmAppShellAssets();
    await warmBuildManifestAssets();
    await warmAssets(DEFAULT_WARM_VENDOR_URLS);
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
      await warmAppShellAssets();
      await warmBuildManifestAssets();
      await warmAssets(DEFAULT_WARM_VENDOR_URLS);
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
  const url = new URL(request.url);
  if (request.mode === "navigate" && url.origin === self.location.origin) {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE_NAME);
        try {
          const networkResponse = await fetch(request);
          if (networkResponse && networkResponse.ok) {
            cache.put(request, networkResponse.clone()).catch(() => {});
            return networkResponse;
          }
        } catch {}
        const cachedShell = await cache.match(new URL("./index.html", APP_BASE_URL).toString());
        if (cachedShell) {
          return cachedShell;
        }
        const cachedRoot = await cache.match(new URL("./", APP_BASE_URL).toString());
        if (cachedRoot) {
          return cachedRoot;
        }
        return fetch(request);
      })(),
    );
    return;
  }
  event.respondWith(cacheFirst(request));
});
