import i18next from "i18next";

export const APP_LANGUAGE_STORAGE_KEY = "ohmymelody.app.language.v1";

export const SUPPORTED_LANGUAGES = ["zh-CN", "en"] as const;

export type AppLanguage = (typeof SUPPORTED_LANGUAGES)[number];

const resources = {
  "zh-CN": {
    translation: {
      app: {
        title: "Oh My Melody",
      },
      language: {
        label: "语言",
        options: {
          "zh-CN": "简体中文",
          en: "English",
        },
      },
      sections: {
        model: "模型",
        audioInput: "音频输入",
        playback: "播放控制",
        display: "显示控制",
        debug: "调试信息",
        console: "控制台日志",
        spectrum: "音频与频谱",
      },
      controls: {
        forceRefresh: "强制重新计算",
        chooseAudio: "选择音频文件",
        dragAudio: "也可以把音频文件拖到这里。",
        startRecording: "开始录制",
        stopRecording: "停止录制",
        exportRecording: "导出录音",
        notRecorded: "未录制",
        recording: "录音中...",
        recorded: "已录制: {{name}}",
      },
      playback: {
        speed: "歌曲速度",
        enableLoop: "启用 A/B 循环",
        segmentA: "A 段",
        segmentB: "B 段",
        setA: "设为 A",
        setB: "设为 B",
        clearLoop: "清除循环",
        shortcutHint: "快捷键：Alt+A 设 A，Alt+B 设 B，Alt+L 开关循环，Alt+Shift+C 清除 A/B。",
        notSet: "未设置",
      },
      display: {
        main: "主视图",
        overview: "总览",
        mainFps: "主视图刷新率",
        mainOverlayFps: "主视图覆盖层刷新率",
        overviewFps: "总览刷新率",
        overviewOverlayFps: "总览覆盖层刷新率",
        representativeMode: "预测策略",
        minUps: "最小 UPS",
        maxUps: "最大 UPS",
        minNote: "最低音高",
        maxNote: "最高音高",
        unlimited: "无限制",
        first: "第一个",
        last: "最后一个",
        firstValid: "第一个有效值",
        lastValid: "最后一个有效值",
        highestConfidence: "最高置信度",
      },
      debug: {
        title: "调试信息",
        enablePanel: "启用调试面板",
        waiting: "等待频谱数据...",
        panelOff: "调试面板已关闭",
      },
      console: {
        description: "保留浏览器控制台输出，同时显示最近 100 条日志。",
      },
      drop: {
        title: "松开文件即可导入",
        text: "把音频文件拖到这个窗口里。",
      },
      status: {
        idle: "空闲",
        running: "运行中",
        succeeded: "已完成",
        failed: "失败",
        cancelled: "已取消",
        disposed: "已释放",
        analyzing: "正在分析音频...",
        cancelledMessage: "已取消",
        failedMessage: "分析失败",
      },
      logs: {
        pageStarted: "页面已启动",
        pwaWarmFailed: "PWA 预热失败: {{message}}",
        pwaRegisterFailed: "PWA 资源注册失败: {{message}}",
        audioLoaded: "音频文件已装载: {{name}}",
        aSet: "A 段已设置: {{time}}",
        bSet: "B 段已设置: {{time}}",
        loopOn: "A/B 循环已开启",
        loopOff: "A/B 循环已关闭",
        loopCleared: "A/B 循环已清除",
        analysisCompleted: "分析完成: {{name}}",
        analysisFailed: "分析失败: {{message}}",
        micAnalysisStopped: "麦克风流式分析已停止",
        micAnalysisRestarted: "麦克风流式分析已重启",
        streamAnalysisCompleted: "流式分析完成",
        streamAnalysisFailed: "流式分析失败: {{message}}",
        micUnsupported: "当前环境不支持麦克风录制",
        micRecorded: "麦克风录音完成: {{name}}",
        micRecordingStarted: "开始麦克风录制",
        micRecordingFailed: "麦克风录制失败: {{message}}",
        micRecordStoppedReason: "麦克风录音已停止",
        noMicRecording: "没有可导出的麦克风录音",
        micExported: "麦克风录音已导出: {{name}}",
        cfpUpdated: "CFP 已更新: batches={{count}}",
        inferenceUpdated: "推理已更新: batchCount={{count}}",
        uiStatus: "UI 状态: {{status}}{{message}}",
        currentRunningIgnoreSwitch: "当前正在分析中，模型切换已忽略",
        modelSwitchedNoFile: "模型已切换，当前没有可重推的音频",
        modelSwitchReanalyze: "模型已切换，开始重新推理: {{name}}",
        dragImport: "拖拽导入: {{name}}",
      },
    },
  },
  en: {
    translation: {
      app: {
        title: "Oh My Melody",
      },
      language: {
        label: "Language",
        options: {
          "zh-CN": "Simplified Chinese",
          en: "English",
        },
      },
      sections: {
        model: "Model",
        audioInput: "Audio Input",
        playback: "Playback",
        display: "Display",
        debug: "Debug",
        console: "Console Log",
        spectrum: "Audio & Spectrum",
      },
      controls: {
        forceRefresh: "Force recompute",
        chooseAudio: "Choose audio file",
        dragAudio: "You can also drag an audio file here.",
        startRecording: "Start recording",
        stopRecording: "Stop recording",
        exportRecording: "Export recording",
        notRecorded: "Not recorded",
        recording: "Recording...",
        recorded: "Recorded: {{name}}",
      },
      playback: {
        speed: "Playback speed",
        enableLoop: "Enable A/B loop",
        segmentA: "A point",
        segmentB: "B point",
        setA: "Set A",
        setB: "Set B",
        clearLoop: "Clear loop",
        shortcutHint: "Shortcuts: Alt+A set A, Alt+B set B, Alt+L toggle loop, Alt+Shift+C clear A/B.",
        notSet: "Not set",
      },
      display: {
        main: "Main",
        overview: "Overview",
        mainFps: "Main refresh rate",
        mainOverlayFps: "Main overlay refresh rate",
        overviewFps: "Overview refresh rate",
        overviewOverlayFps: "Overview overlay refresh rate",
        representativeMode: "Representative mode",
        minUps: "Min UPS",
        maxUps: "Max UPS",
        minNote: "Lowest pitch",
        maxNote: "Highest pitch",
        unlimited: "Unlimited",
        first: "First",
        last: "Last",
        firstValid: "First valid",
        lastValid: "Last valid",
        highestConfidence: "Highest confidence",
      },
      debug: {
        title: "Debug",
        enablePanel: "Enable debug panel",
        waiting: "Waiting for spectrum data...",
        panelOff: "Debug panel is off",
      },
      console: {
        description: "Keeps browser console output and mirrors the latest 100 log entries here.",
      },
      drop: {
        title: "Release to import",
        text: "Drop the audio file into this window.",
      },
      status: {
        idle: "idle",
        running: "running",
        succeeded: "succeeded",
        failed: "failed",
        cancelled: "cancelled",
        disposed: "disposed",
        analyzing: "Analyzing audio...",
        cancelledMessage: "Cancelled",
        failedMessage: "Analysis failed",
      },
      logs: {
        pageStarted: "Page started",
        pwaWarmFailed: "PWA warm-up failed: {{message}}",
        pwaRegisterFailed: "PWA asset registration failed: {{message}}",
        audioLoaded: "Audio file loaded: {{name}}",
        aSet: "A point set: {{time}}",
        bSet: "B point set: {{time}}",
        loopOn: "A/B loop enabled",
        loopOff: "A/B loop disabled",
        loopCleared: "A/B loop cleared",
        analysisCompleted: "Analysis completed: {{name}}",
        analysisFailed: "Analysis failed: {{message}}",
        micAnalysisStopped: "Microphone stream analysis stopped",
        micAnalysisRestarted: "Microphone stream analysis restarted",
        streamAnalysisCompleted: "Stream analysis completed",
        streamAnalysisFailed: "Stream analysis failed: {{message}}",
        micUnsupported: "This environment does not support microphone recording",
        micRecorded: "Microphone recording completed: {{name}}",
        micRecordingStarted: "Started microphone recording",
        micRecordingFailed: "Microphone recording failed: {{message}}",
        micRecordStoppedReason: "Microphone recording stopped",
        noMicRecording: "No microphone recording to export",
        micExported: "Microphone recording exported: {{name}}",
        cfpUpdated: "CFP updated: batches={{count}}",
        inferenceUpdated: "Inference updated: batchCount={{count}}",
        uiStatus: "UI status: {{status}}{{message}}",
        currentRunningIgnoreSwitch: "Currently analyzing, model switch ignored",
        modelSwitchedNoFile: "Model switched, no audio to re-run",
        modelSwitchReanalyze: "Model switched, re-running: {{name}}",
        dragImport: "Drag import: {{name}}",
      },
    },
  },
} as const;

function normalizeLanguage(value: string | null | undefined): AppLanguage {
  return value === "zh-CN" ? "zh-CN" : "en";
}

function readStoredLanguage(): AppLanguage | null {
  if (typeof localStorage === "undefined") {
    return null;
  }
  try {
    const stored = localStorage.getItem(APP_LANGUAGE_STORAGE_KEY);
    return stored ? normalizeLanguage(stored) : null;
  } catch {
    return null;
  }
}

export function detectAppLanguage(): AppLanguage {
  const stored = readStoredLanguage();
  if (stored) {
    return stored;
  }
  if (typeof navigator === "undefined") {
    return "en";
  }
  const language = normalizeLanguage(navigator.language);
  if (language === "zh-CN") {
    return language;
  }
  const languages = Array.isArray(navigator.languages) ? navigator.languages : [];
  const preferred = languages.find((item) => normalizeLanguage(item) === "zh-CN");
  return preferred ? "zh-CN" : "en";
}

export async function initAppI18n(): Promise<void> {
  await i18next.init({
    lng: detectAppLanguage(),
    fallbackLng: "en",
    resources,
    interpolation: {
      escapeValue: false,
    },
    returnNull: false,
    returnEmptyString: false,
  });
  if (typeof document !== "undefined") {
    document.documentElement.lang = i18next.language;
  }
}

export function getAppLanguage(): AppLanguage {
  return normalizeLanguage(i18next.language);
}

export async function setAppLanguage(language: string): Promise<AppLanguage> {
  const next = normalizeLanguage(language);
  if (typeof localStorage !== "undefined") {
    try {
      localStorage.setItem(APP_LANGUAGE_STORAGE_KEY, next);
    } catch {}
  }
  await i18next.changeLanguage(next);
  if (typeof document !== "undefined") {
    document.documentElement.lang = next;
  }
  return next;
}

export function t(key: string, values?: Record<string, unknown>): string {
  return values ? String(i18next.t(key, values)) : String(i18next.t(key));
}

export function applyLocalizedText(root?: ParentNode | null): void {
  const scope = root ?? (typeof document !== "undefined" ? document : null);
  if (!scope) {
    return;
  }
  const elements = scope.querySelectorAll<HTMLElement>(
    "[data-i18n], [data-i18n-value], [data-i18n-title], [data-i18n-aria-label], [data-i18n-placeholder]",
  );
  for (const element of elements) {
    const textKey = element.getAttribute("data-i18n");
    if (textKey) {
      element.textContent = t(textKey);
    }
    const valueKey = element.getAttribute("data-i18n-value");
    if (valueKey && "value" in element) {
      (element as HTMLInputElement | HTMLOptionElement | HTMLTextAreaElement | HTMLSelectElement).value = t(valueKey);
    }
    const titleKey = element.getAttribute("data-i18n-title");
    if (titleKey) {
      element.setAttribute("title", t(titleKey));
    }
    const ariaLabelKey = element.getAttribute("data-i18n-aria-label");
    if (ariaLabelKey) {
      element.setAttribute("aria-label", t(ariaLabelKey));
    }
    const placeholderKey = element.getAttribute("data-i18n-placeholder");
    if (placeholderKey && "placeholder" in element) {
      (element as HTMLInputElement | HTMLTextAreaElement).placeholder = t(placeholderKey);
    }
  }
}
