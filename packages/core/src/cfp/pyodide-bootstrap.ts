import { isLikelyHtmlText } from "../script/script.js";
import { getPyodidePathAppendPython } from "../script/script.js";

export interface PyodideLike {
  loadPackage: (packages: string[]) => Promise<unknown>;
  runPython: (code: string) => unknown;
  runPythonAsync: (code: string) => Promise<unknown>;
  globals: {
    set(name: string, value: unknown): void;
  };
  FS: {
    writeFile(path: string, code: string): void;
    readFile(path: string): Uint8Array;
    unlink?: (path: string) => void;
  } | null;
  toPy: (value: Float32Array) => { destroy?: () => void } | unknown;
}

export interface CFPBootstrapEnvironment {
  loadScript: (scriptUrl: string) => Promise<unknown>;
  loadPyodide: (options: { indexURL: string }) => Promise<PyodideLike>;
}

async function loadPythonSourceFromUrl(url: string): Promise<string> {
  if (!url) {
    return "";
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch cfp.py (HTTP ${response.status}). Ensure cfp.py is served from the core package path.`);
  }
  const code = await response.text();
  if (isLikelyHtmlText(code)) {
    throw new Error("cfp.py fetch returned HTML (likely a 404 page). Ensure cfp.py is present and served correctly.");
  }
  return code;
}

export async function fetchCFPScriptSource(cfpScriptUrl: string): Promise<string> {
  return await loadPythonSourceFromUrl(cfpScriptUrl);
}

export async function installCFPScriptIntoPyodide(
  pyodide: PyodideLike,
  code: string,
): Promise<boolean> {
  if (!pyodide || !pyodide.FS || typeof pyodide.FS.writeFile !== "function") {
    throw new Error("pyodide FS is unavailable");
  }
  pyodide.FS.writeFile("cfp.py", code);
  if (typeof pyodide.runPython !== "function") {
    throw new Error("pyodide runPython is unavailable");
  }
  pyodide.runPython(getPyodidePathAppendPython());
  return true;
}

export async function initializePyodideForCFP({
  pyodideScriptUrl,
  pyodideIndexURL,
  cfpScriptUrl,
  packages = ["numpy", "scipy", "pandas"],
  environment,
}: {
  pyodideScriptUrl: string;
  pyodideIndexURL: string;
  cfpScriptUrl: string;
  packages?: string[];
  environment: CFPBootstrapEnvironment;
}): Promise<PyodideLike> {
  if (!environment || typeof environment.loadScript !== "function" || typeof environment.loadPyodide !== "function") {
    throw new Error("CFP bootstrap environment is unavailable");
  }

  await environment.loadScript(pyodideScriptUrl);

  const pyodide = await environment.loadPyodide({
    indexURL: pyodideIndexURL,
  });

  if (Array.isArray(packages) && packages.length && typeof pyodide.loadPackage === "function") {
    await pyodide.loadPackage(packages);
  }

  const code = await fetchCFPScriptSource(cfpScriptUrl);
  await installCFPScriptIntoPyodide(pyodide, code);
  return pyodide;
}
