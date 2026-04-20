export type RafTaskResult = void | null | Promise<unknown>;

export type RafTaskFn = () => RafTaskResult;

export interface RafTaskOptions {
  priority?: number;
  targetFps?: number | null;
}

export interface RafScheduledTask {
  id: string;
  overdueMs: number;
  taskState: {
    priority: number;
    targetFps: number | null;
    nextRunTs: number;
    runCount: number;
    lastRunTs: number;
    intervalMs: number;
  };
}

export type RafTaskComparator = (left: RafScheduledTask, right: RafScheduledTask) => number;

interface RafTaskState {
  fn: RafTaskFn;
  priority: number;
  targetFps: number | null;
  intervalMs: number;
  nextRunTs: number;
  runCount: number;
  lastRunTs: number;
  recentRunTimes: number[];
}

export interface RafSchedulerStats {
  taskCount: number;
  fps: number;
  estimatedFps: number;
  frameBudgetMs: number;
  tasks: Array<{
    id: string;
    priority: number;
    targetFps: number;
    actualFps: number;
    nextRunInMs: number;
    runCount: number;
  }>;
}

export interface RafTaskSnapshot {
  id: string;
  priority: number;
  targetFps: number | null;
  intervalMs: number;
  nextRunTs: number;
  runCount: number;
  lastRunTs: number;
  recentRunTimes: number[];
}

export interface RafSchedulerOptions {
  compareTasks?: RafTaskComparator | null;
  now?: () => number;
}

export class RafScheduler {
  private tasks = new Map<string, RafTaskState>();
  private compareTasks: RafTaskComparator;
  private nextId = 1;
  private rafId: number | null = null;
  private running = false;
  private recentRafTimes: number[] = [];
  private readonly nowFn: () => number;

  constructor(options: RafSchedulerOptions = {}) {
    this.nowFn =
      options.now ??
      (() => (typeof performance !== 'undefined' && performance.now ? performance.now() : Date.now()));
    this.compareTasks =
      typeof options.compareTasks === 'function'
        ? options.compareTasks
        : (left, right) => {
            const leftPriority = this.getEffectivePriority(left.taskState.priority, left.overdueMs);
            const rightPriority = this.getEffectivePriority(right.taskState.priority, right.overdueMs);
            if (leftPriority !== rightPriority) {
              return leftPriority - rightPriority;
            }
            if (left.taskState.nextRunTs !== right.taskState.nextRunTs) {
              return left.taskState.nextRunTs - right.taskState.nextRunTs;
            }
            return Number(left.id) - Number(right.id);
          };
  }

  private now(): number {
    return this.nowFn();
  }

  private normalizeIntervalMs(targetFps: number | null | undefined): number {
    if (!Number.isFinite(targetFps) || Number(targetFps) <= 0) {
      return 0;
    }
    return 1000 / Number(targetFps);
  }

  private getEstimatedRafFps(): number {
    return Math.max(1, this.recentRafTimes.length || 60);
  }

  private getDynamicFrameBudgetMs(): number {
    const estimatedFps = this.getEstimatedRafFps();
    const frameIntervalMs = 1000 / estimatedFps;
    const safetyMs = 3;
    return Math.max(4, frameIntervalMs - safetyMs);
  }

  private getEffectivePriority(priority: number, overdueMs: number): number {
    const estimatedFps = this.getEstimatedRafFps();
    const frameIntervalMs = 1000 / estimatedFps;
    const agingBoost = Math.min(8, Math.floor(overdueMs / frameIntervalMs));
    return priority - agingBoost;
  }

  addTask(
    fn: RafTaskFn,
    { priority = 10, targetFps = null }: RafTaskOptions = {},
  ): string {
    const id = String(this.nextId++);
    const now = this.now();
    const intervalMs = this.normalizeIntervalMs(targetFps);
    this.tasks.set(id, {
      fn,
      priority,
      targetFps: Number.isFinite(targetFps) && Number(targetFps) > 0 ? Number(targetFps) : null,
      intervalMs,
      nextRunTs: now,
      runCount: 0,
      lastRunTs: 0,
      recentRunTimes: [],
    });
    return id;
  }

  updateTask(id: string, options: RafTaskOptions = {}): boolean {
    const task = this.tasks.get(String(id));
    if (!task) {
      return false;
    }

    const now = this.now();
    if (Object.prototype.hasOwnProperty.call(options, 'priority')) {
      const nextPriority = Number(options.priority);
      if (Number.isFinite(nextPriority)) {
        task.priority = nextPriority;
      }
    }
    if (Object.prototype.hasOwnProperty.call(options, 'targetFps')) {
      const nextTargetFps =
        Number.isFinite(options.targetFps) && Number(options.targetFps) > 0
          ? Number(options.targetFps)
          : null;
      task.targetFps = nextTargetFps;
      task.intervalMs = this.normalizeIntervalMs(nextTargetFps);
      task.nextRunTs = now;
    }
    return true;
  }

  removeTask(id: string): void {
    this.tasks.delete(String(id));
  }

  start(): void {
    if (this.running) {
      return;
    }

    this.running = true;
    const loop = () => {
      if (!this.running) {
        return;
      }

      const now = this.now();
      this.recentRafTimes.push(now);
      const cutoff = now - 1000;
      while (this.recentRafTimes.length && (this.recentRafTimes[0] ?? 0) < cutoff) {
        this.recentRafTimes.shift();
      }

      this.runFrame(now);

      if (typeof requestAnimationFrame === 'function') {
        this.rafId = requestAnimationFrame(loop);
        return;
      }

      this.rafId = setTimeout(loop, 16);
    };

    if (typeof requestAnimationFrame === 'function') {
      this.rafId = requestAnimationFrame(loop);
      return;
    }

    this.rafId = setTimeout(loop, 16);
  }

  stop(): void {
    if (this.rafId !== null) {
      if (typeof cancelAnimationFrame === 'function') {
        cancelAnimationFrame(this.rafId);
      } else {
        clearTimeout(this.rafId);
      }
    }
    this.rafId = null;
    this.running = false;
  }

  private runFrame(now: number): void {
    const frameStart = now;
    const frameBudgetMs = this.getDynamicFrameBudgetMs();
    const dueTasks = Array.from(this.tasks.entries())
      .filter(([_, taskState]) => (taskState.nextRunTs || 0) <= now)
      .map(([id, taskState]) => ({
        id,
        overdueMs: Math.max(0, now - taskState.nextRunTs),
        taskState,
      }));

    dueTasks.sort((left, right) => this.compareTasks(left, right));

    for (let index = 0; index < dueTasks.length; index += 1) {
      if (this.now() - frameStart >= frameBudgetMs) {
        break;
      }

      const dueTask = dueTasks[index];
      if (!dueTask) {
        continue;
      }
      const task = this.tasks.get(dueTask.id);
      if (!task) {
        continue;
      }

      try {
        task.fn();
        task.runCount += 1;
        task.lastRunTs = now;
        task.recentRunTimes.push(now);
        const cutoff = now - 1000;
        while (task.recentRunTimes.length && (task.recentRunTimes[0] ?? 0) < cutoff) {
          task.recentRunTimes.shift();
        }

        if (task.targetFps && task.targetFps > 0) {
          task.intervalMs = this.normalizeIntervalMs(task.targetFps);
          task.nextRunTs = now + task.intervalMs;
        } else {
          this.tasks.delete(dueTask.id);
        }
      } catch {
        this.tasks.delete(dueTask.id);
      }
    }
  }

  getStats(): RafSchedulerStats {
    const now = this.now();
    const fps = this.recentRafTimes.length;
    const estimatedFps = this.getEstimatedRafFps();
    const frameBudgetMs = this.getDynamicFrameBudgetMs();
    const tasks = Array.from(this.tasks.entries()).map(([id, task]) => ({
      id,
      priority: task.priority,
      targetFps: task.targetFps || 0,
      actualFps: task.recentRunTimes.length,
      nextRunInMs: Math.max(0, (task.nextRunTs || 0) - now),
      runCount: task.runCount,
    }));

    return {
      taskCount: this.tasks.size,
      fps,
      estimatedFps,
      frameBudgetMs,
      tasks,
    };
  }

  getTaskSnapshot(id: string): RafTaskSnapshot | null {
    const task = this.tasks.get(String(id));
    if (!task) {
      return null;
    }
    return {
      id: String(id),
      priority: task.priority,
      targetFps: task.targetFps,
      intervalMs: task.intervalMs,
      nextRunTs: task.nextRunTs,
      runCount: task.runCount,
      lastRunTs: task.lastRunTs,
      recentRunTimes: [...task.recentRunTimes],
    };
  }
}
