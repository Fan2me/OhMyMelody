export const CORE_MODEL_NAMES = [
  "mamba_a.onnx",
  "mamba_b.onnx",
  "mftfa_a.onnx",
  "mftfa_b.onnx",
  "msnet.onnx",
] as const;

export const CORE_MODEL_DEFAULT_NAME = "msnet.onnx";

export type CoreModelName = (typeof CORE_MODEL_NAMES)[number];

export interface CoreModelCatalogEntry {
  name: CoreModelName;
  url: string;
}

export function getCoreModelNames(): readonly CoreModelName[] {
  return CORE_MODEL_NAMES;
}

export function isCoreModelName(value: unknown): value is CoreModelName {
  return (
    typeof value === "string" &&
    CORE_MODEL_NAMES.includes(value as CoreModelName)
  );
}

export function normalizeCoreModelName(value: unknown): CoreModelName {
  if (isCoreModelName(value)) {
    return value;
  }
  return CORE_MODEL_DEFAULT_NAME;
}

export function resolveCoreModelUrl(
  modelName: string | null | undefined,
): string {
  const safeName = normalizeCoreModelName(modelName);
  return new URL(
    `../models/${encodeURIComponent(safeName)}`,
    import.meta.url,
  ).toString();
}

export function resolveCoreModelManifestUrl(): string {
  return new URL("../models/models.json", import.meta.url).toString();
}

export function getCoreModelCatalog(): CoreModelCatalogEntry[] {
  return CORE_MODEL_NAMES.map((name) => ({
    name,
    url: resolveCoreModelUrl(name),
  }));
}
