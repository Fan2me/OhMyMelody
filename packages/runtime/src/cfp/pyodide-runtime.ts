import {
  type CFPBootstrapEnvironment,
  initializePyodideForCFP,
  type PyodideLike,
} from "@ohm/core/cfp/pyodide-bootstrap.js";
import type { PyodideWorkerLike } from "@ohm/core/cfp/types.js";

export type { PyodideWorkerLike } from "@ohm/core/cfp/types.js";

export async function loadBrowserPyodideRuntime(
  pyodideScriptUrl: string,
  indexURL: string,
): Promise<PyodideLike> {
  const runtimeGlobal = globalThis as typeof globalThis & {
    loadPyodide?: (options: { indexURL: string }) => Promise<PyodideLike>;
  };

  if (typeof runtimeGlobal.loadPyodide !== "function") {
    if (typeof document === "undefined") {
      throw new Error("Pyodide is not available in current environment");
    }

    await new Promise<void>((resolve, reject) => {
      const existingScript = document.querySelector<HTMLScriptElement>(
        'script[data-ohm-pyodide="true"]',
      );
      const finish = () => resolve();
      const fail = () => reject(new Error("failed to load pyodide runtime"));

      if (existingScript) {
        if (typeof runtimeGlobal.loadPyodide === "function") {
          resolve();
          return;
        }
        existingScript.addEventListener("load", finish, { once: true });
        existingScript.addEventListener("error", fail, { once: true });
        return;
      }

      const script = document.createElement("script");
      script.dataset.ohmPyodide = "true";
      script.src = pyodideScriptUrl;
      script.async = true;
      script.onload = finish;
      script.onerror = fail;
      document.head.appendChild(script);
    });
  }

  const loadPyodide = runtimeGlobal.loadPyodide?.bind(runtimeGlobal);
  if (!loadPyodide) {
    throw new Error("Pyodide is not available in current environment");
  }
  return loadPyodide({ indexURL });
}

export async function initializeBrowserCFPPyodide({
  pyodideScriptUrl,
  pyodideIndexURL,
  cfpScriptUrl,
}: {
  pyodideScriptUrl: string;
  pyodideIndexURL: string;
  cfpScriptUrl: string;
}): Promise<PyodideWorkerLike> {
  const browserPyodide = await loadBrowserPyodideRuntime(
    pyodideScriptUrl,
    pyodideIndexURL,
  );

  return (await initializePyodideForCFP({
    pyodideScriptUrl,
    pyodideIndexURL,
    cfpScriptUrl,
    packages: ["numpy", "scipy"],
    environment: {
      loadScript: async () => true,
      loadPyodide: async () => browserPyodide,
    } satisfies CFPBootstrapEnvironment,
  })) as PyodideWorkerLike;
}
