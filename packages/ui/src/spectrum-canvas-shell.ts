export interface SpectrumCanvasShellPartState {
  enabled: boolean;
  overlay: boolean;
  maxFps: number;
}

export interface SpectrumCanvasShellPartConfigObject {
  enabled?: boolean;
  overlay?: boolean;
  maxFps?: number;
}

export type SpectrumCanvasShellPartConfigValue =
  | boolean
  | SpectrumCanvasShellPartConfigObject
  | undefined;

export interface SpectrumCanvasShellPartsConfig {
  main?: SpectrumCanvasShellPartConfigValue;
  overview?: SpectrumCanvasShellPartConfigValue;
}

export interface SpectrumCanvasShellOptions {
  parts?: SpectrumCanvasShellPartsConfig;
}

export interface SpectrumCanvasShellPartRef {
  section: HTMLElement;
  base: HTMLCanvasElement;
  overlay: HTMLCanvasElement;
  baseCtx: CanvasRenderingContext2D | null;
  overlayCtx: CanvasRenderingContext2D | null;
}

export function createSpectrumSection(
  documentRef: Document,
  title: string,
  minHeight: number,
): SpectrumCanvasShellPartRef {
  const section = documentRef.createElement("section");
  section.dataset.spectrumSection = title;
  section.style.position = "relative";
  section.style.width = "100%";
  section.style.minHeight = `${minHeight}px`;
  section.style.overflow = "hidden";
  section.style.border = "1px solid rgba(54, 80, 130, 0.08)";
  section.style.borderRadius = "0";
  section.style.background = "rgba(245, 248, 255, 0.9)";

  const base = documentRef.createElement("canvas");
  const overlay = documentRef.createElement("canvas");
  for (const canvas of [base, overlay]) {
    canvas.style.position = "absolute";
    canvas.style.inset = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
  }
  overlay.style.pointerEvents = "none";
  section.appendChild(base);
  section.appendChild(overlay);

  return {
    section,
    base,
    overlay,
    baseCtx: base.getContext("2d"),
    overlayCtx:
      overlay.getContext("2d", { alpha: true, desynchronized: true }) ||
      overlay.getContext("2d"),
  };
}

export function createSpectrumRootContainer(documentRef: Document): HTMLDivElement {
  const root = documentRef.createElement("div");
  root.style.display = "flex";
  root.style.flexDirection = "column";
  root.style.gap = "0";
  root.style.position = "relative";
  root.style.width = "100%";
  root.style.height = "100%";
  root.style.boxSizing = "border-box";
  root.style.padding = "12px";
  root.style.background = "transparent";
  return root;
}
