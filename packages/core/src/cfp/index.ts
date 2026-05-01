export * from "./cfp.js";
export * from "./chunk.js";
export * from "./common.js";
export * from "./pyodide-bootstrap.js";
export * from "./worker-manager.js";
export * from "./types.js";

export const CORE_CFP_SCRIPT_URL = new URL(
  "../../cfp.py",
  import.meta.url,
).toString();

export const CORE_CFP_WORKER_MODULE_URL = new URL(
  "./worker.js?worker",
  import.meta.url,
).toString();
