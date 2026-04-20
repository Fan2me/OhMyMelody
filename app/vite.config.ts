import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { copyFile, mkdir } from "node:fs/promises";
import { defineConfig } from "vite";

const appRoot = fileURLToPath(new URL(".", import.meta.url));
const workspaceRoot = resolve(appRoot, "..");

function appRuntimeAssetsPlugin() {
  const appSwSource = resolve(workspaceRoot, "app/sw.js");
  const appSwDist = resolve(appRoot, "dist/sw.js");
  const appFaviconSource = resolve(appRoot, "favicon.ico");
  const appFaviconDist = resolve(appRoot, "dist/favicon.ico");

  async function copyFileSafe(sourcePath: string, destinationPath: string): Promise<void> {
    await mkdir(resolve(destinationPath, ".."), { recursive: true });
    await copyFile(sourcePath, destinationPath);
  }

  return {
    name: "app-runtime-assets",
    apply: "build" as const,
    transformIndexHtml() {
      return {
        tags: [
          {
            tag: "script",
            attrs: { type: "importmap" },
            injectTo: "head-prepend",
            children: JSON.stringify(
              {
                imports: {
                  "@ohm/core": "../../packages/core/dist/index.js",
                  "@ohm/core/": "../../packages/core/dist/",
                  "@ohm/runtime": "../../packages/runtime/dist/index.js",
                  "@ohm/runtime/": "../../packages/runtime/dist/",
                  "@ohm/ui": "../../packages/ui/dist/index.js",
                  "@ohm/ui/": "../../packages/ui/dist/",
                },
              },
              null,
              2,
            ),
          },
        ],
      };
    },
    async closeBundle() {
      await copyFileSafe(appFaviconSource, appFaviconDist);
      try {
        await copyFileSafe(appSwSource, appSwDist);
      } catch {}
    },
  };
}

export default defineConfig({
  root: appRoot,
  base: "./",
  publicDir: false,
  optimizeDeps: {
    exclude: ["@ohm/core", "@ohm/runtime", "@ohm/ui"],
  },
  server: {
    fs: {
      allow: [workspaceRoot],
    },
  },
  plugins: [appRuntimeAssetsPlugin()],
  build: {
    emptyOutDir: true,
    outDir: resolve(appRoot, "dist"),
    assetsInlineLimit: 0,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "ohm-core",
              test: /[\\/]packages[\\/]core[\\/]/,
            },
            {
              name: "ohm-runtime",
              test: /[\\/]packages[\\/]runtime[\\/]/,
            },
            {
              name: "ohm-ui",
              test: /[\\/]packages[\\/]ui[\\/]/,
            },
            {
              name: "vendor",
              test: /[\\/]node_modules[\\/]/,
            },
          ],
        },
        entryFileNames: "app.js",
        chunkFileNames: "chunks/[name]-[hash].js",
        assetFileNames: "assets/[name]-[hash][extname]",
      },
    },
  },
});
