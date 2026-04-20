import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { spawn } from 'node:child_process';

const rootDir = process.cwd();
const browserCandidates = [
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
];

function pickBrowserPath() {
  for (const candidate of browserCandidates) {
    if (candidate) {
      return candidate;
    }
  }
  throw new Error('Microsoft Edge executable was not found');
}

function getContentType(filePath) {
  if (filePath.endsWith('.html')) return 'text/html; charset=utf-8';
  if (filePath.endsWith('.js') || filePath.endsWith('.mjs')) return 'text/javascript; charset=utf-8';
  if (filePath.endsWith('.json')) return 'application/json; charset=utf-8';
  if (filePath.endsWith('.css')) return 'text/css; charset=utf-8';
  return 'application/octet-stream';
}

function createFileServer(baseDir) {
  return createServer(async (req, res) => {
    try {
      const url = new URL(req.url || '/', 'http://127.0.0.1');
      const pathname = decodeURIComponent(url.pathname);
      const relPath = pathname === '/' ? '/benchmarks/audio-capture-benchmark.html' : pathname;
      const filePath = path.join(baseDir, relPath);
      const fullPath = path.resolve(filePath);
      if (!fullPath.startsWith(path.resolve(baseDir))) {
        res.writeHead(403, { 'content-type': 'text/plain; charset=utf-8' });
        res.end('forbidden');
        return;
      }

      const fileStat = await stat(fullPath);
      if (!fileStat.isFile()) {
        res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
        res.end('not found');
        return;
      }

      const body = await readFile(fullPath);
      res.writeHead(200, {
        'content-type': getContentType(fullPath),
        'cache-control': 'no-store',
      });
      res.end(body);
    } catch {
      res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
      res.end('not found');
    }
  });
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function waitFor(predicate, timeoutMs, intervalMs = 100) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const value = await predicate();
    if (value) {
      return value;
    }
    await wait(intervalMs);
  }
  throw new Error('Timed out while waiting for benchmark result');
}

async function connectToBrowser(port) {
  const version = await waitFor(async () => {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/json/version`);
      if (!response.ok) {
        return null;
      }
      return await response.json();
    } catch {
      return null;
    }
  }, 15000);

  const socket = new WebSocket(version.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });

  let nextId = 1;
  const pending = new Map();

  socket.addEventListener('message', (event) => {
    const message = JSON.parse(event.data);
    if (message.id && pending.has(message.id)) {
      const entry = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) {
        entry.reject(new Error(message.error.message || 'CDP command failed'));
        return;
      }
      entry.resolve(message.result);
    }
  });

  function send(method, params = {}, sessionId) {
    const id = nextId += 1;
    const payload = { id, method, params };
    if (sessionId) {
      payload.sessionId = sessionId;
    }
    socket.send(JSON.stringify(payload));
    return new Promise((resolve, reject) => {
      pending.set(id, { resolve, reject });
    });
  }

  return { socket, send };
}

async function run() {
  const server = createFileServer(rootDir);
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  if (!address || typeof address !== 'object') {
    throw new Error('Failed to start benchmark server');
  }

  const browserPath = pickBrowserPath();
  const userDataDir = path.join(os.tmpdir(), `ohm-audio-benchmark-${Date.now()}`);
  const browser = spawn(
    browserPath,
    [
      '--headless=new',
      '--disable-gpu',
      '--autoplay-policy=no-user-gesture-required',
      '--disable-background-networking',
      '--disable-extensions',
      '--no-first-run',
      '--no-default-browser-check',
      `--user-data-dir=${userDataDir}`,
      `--remote-debugging-port=${9222}`,
      'about:blank',
    ],
    { stdio: 'ignore' },
  );

  try {
    const { send, socket } = await connectToBrowser(9222);
    const target = await send('Target.createTarget', { url: 'about:blank' });
    const session = await send('Target.attachToTarget', { targetId: target.targetId, flatten: true });
    const sessionId = session.sessionId;

    await send('Page.enable', {}, sessionId);
    await send('Runtime.enable', {}, sessionId);
    await send('Page.navigate', {
      url: `http://127.0.0.1:${address.port}/benchmarks/audio-capture-benchmark.html`,
    }, sessionId);

    const result = await waitFor(async () => {
      const evalResult = await send('Runtime.evaluate', {
        expression: 'window.__OHM_BENCH_RESULT__ || null',
        returnByValue: true,
        awaitPromise: true,
      }, sessionId);
      return evalResult?.result?.value ?? null;
    }, 120000, 500);

    console.log(JSON.stringify(result, null, 2));

    try {
      await send('Target.closeTarget', { targetId: target.targetId });
    } catch {}
    socket.close();
  } finally {
    browser.kill('SIGTERM');
    server.close();
  }
}

run().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
