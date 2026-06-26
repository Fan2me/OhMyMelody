import assert from "node:assert/strict";
import test from "node:test";

import {
  buildFfmpegAssetUrls,
  FFMPEG_CORE_BASE_URL,
  FFMPEG_CORE_CDN_VERSION,
} from "../src/ffmpeg-transcode.ts";

test("ffmpeg asset urls target the esm core bundle", () => {
  const urls = buildFfmpegAssetUrls();

  assert.equal(
    FFMPEG_CORE_BASE_URL,
    `https://cdn.jsdelivr.net/npm/@ffmpeg/core@${FFMPEG_CORE_CDN_VERSION}/dist/esm`,
  );
  assert.equal(urls.baseURL, FFMPEG_CORE_BASE_URL);
  assert.equal(urls.coreURL, `${FFMPEG_CORE_BASE_URL}/ffmpeg-core.js`);
  assert.equal(urls.wasmURL, `${FFMPEG_CORE_BASE_URL}/ffmpeg-core.wasm`);
  assert.match(urls.coreURL, /\/dist\/esm\/ffmpeg-core\.js$/);
  assert.doesNotMatch(urls.coreURL, /\/dist\/umd\//);
});

test("ffmpeg asset urls can be built from a custom base url", () => {
  const urls = buildFfmpegAssetUrls("https://example.com/ffmpeg");

  assert.deepEqual(urls, {
    baseURL: "https://example.com/ffmpeg",
    coreURL: "https://example.com/ffmpeg/ffmpeg-core.js",
    wasmURL: "https://example.com/ffmpeg/ffmpeg-core.wasm",
  });
});
