# 主频谱拖动红线坐标错位修复

## Status

Phase 4 — in progress

## Problem and goal

主频谱在鼠标点击后拖动时，红色绘制线相对鼠标指针向右偏移，导致视觉上的点击位置与绘制位置不一致。该行为疑似由近期提交引入。目标是恢复主频谱中鼠标位置与红线位置的一致性，同时保持总览频谱滑动窗口拖动的现有行为不变。

## Scope

- 修复页面主频谱的鼠标按下、移动、释放过程中的坐标换算或绘制定位问题。
- 兼容页面滚动、窗口尺寸变化、浏览器缩放及不同设备像素比下的坐标定位。
- 保持总览频谱滑动窗口拖动逻辑和交互行为不变。

## Non-goals

- 不改变红线的颜色、样式、宽度或其他视觉设计。
- 不改变总览频谱滑动窗口的交互规则。
- 不扩展触摸手势、键盘操作或其他输入方式，除非现有实现已共享同一坐标转换接口且修复可自然覆盖。
- 不处理与本问题无关的工作区既有变更。

## Confirmed decisions

- 问题范围仅限页面中的主频谱；总览频谱当前正常。
- 验收要求为：主频谱在鼠标按下、持续拖动和释放的整个过程中，红线与鼠标指针所在的频谱位置重合。
- 需要考虑页面滚动、窗口缩放、浏览器缩放和设备像素比。
- 本次变更应限于坐标错位修复，不调整无关交互或视觉样式。

## Functional requirements

1. 主频谱收到鼠标按下事件时，红线起始位置必须对应事件的实际频谱坐标。
2. 主频谱拖动过程中，红线位置必须跟随鼠标事件对应的频谱坐标，不得出现固定的水平偏移。
3. 页面滚动、窗口尺寸变化、浏览器缩放或设备像素比变化不应引入可观察的系统性偏移。
4. 总览频谱滑动窗口的拖动位置和行为必须保持兼容。
5. 现有非拖动频谱绘制与布局行为必须保持兼容。

## Interfaces and data contracts

- 鼠标事件到主频谱内部坐标的转换必须以实际承载绘制区域的边界、滚动偏移和当前绘制尺寸为依据。
- 若主频谱使用 Canvas 或等效绘制表面，CSS 显示尺寸与内部像素尺寸之间的比例必须在坐标转换中保持一致。
- 总览频谱的坐标转换接口或调用方不得因本修复发生行为变化。

## Error and edge-case behavior

- 当鼠标事件发生在主频谱绘制区域边界时，红线应落在对应边界位置，不应因边界矩形偏移而向右错位。
- 绘制区域经过页面滚动、窗口 resize 或浏览器缩放后，后续拖动应重新基于当前几何信息定位。
- 鼠标拖动在绘制区域外继续移动时，沿用现有的边界限制或结束策略，不新增越界行为。

## Acceptance criteria

- Given 主频谱处于可交互状态，When 在任意横向位置按下鼠标，Then 红线起点与鼠标按下位置重合。
- Given 鼠标已在主频谱上按下，When 向左或向右持续拖动，Then 红线在每个更新位置都与鼠标指针对应位置重合，不能出现固定右偏。
- Given 页面已滚动、窗口尺寸已改变、浏览器缩放比例非 100% 或设备像素比不为 1，When 在主频谱上点击并拖动，Then 红线仍与鼠标位置重合。
- Given 总览频谱滑动窗口当前可拖动，When 执行原有拖动操作，Then 滑动窗口行为与修复前一致。
- Given 项目现有自动化测试与构建流程，When 执行批准的聚焦测试和回归测试，Then 均通过。

## Open questions

无。实现阶段如发现坐标模型或测试环境存在未确认的行为差异，应返回澄清阶段。

## TDD acceptance tests

- 添加或更新主频谱坐标转换的单元测试，覆盖 CSS 尺寸与内部绘制尺寸不一致时的鼠标坐标映射。
- 添加主频谱拖动行为测试，验证按下和连续移动产生的红线坐标与事件坐标一致，并验证不存在固定水平偏移。
- 添加页面滚动、缩放/设备像素比或等价几何偏移场景的回归测试，按项目现有测试能力选择可稳定表达的形式。
- 执行总览频谱拖动相关现有测试，确保其行为未受影响。

## Approval record

- 用户已明确授权进入 Phase 2 和 Phase 3。
- Phase 2 Spec 基于已确认的问题范围、兼容要求和验收标准创建。

## Impact assessment

当前实现存在两条相关坐标链路：

1. 主频谱交互在 `spectrum-interaction.ts` 中根据鼠标事件的 `clientX`、Canvas 的 `getBoundingClientRect()` 和主图宽度计算频谱帧，再通过 `audioElement.currentTime` 驱动红色播放线。
2. 主频谱叠加层在 `spectrum-main-overlay.ts` 中根据 `audioElement.currentTime` 反算播放线位置，同时根据 Canvas 的 CSS 尺寸和内部像素尺寸计算绘制坐标。

两条链路若对频谱帧范围、持续时间、Canvas CSS 尺寸、内部像素尺寸或 DPR 的基准不一致，就会造成红线相对鼠标出现系统性水平偏移。近期提交 `45da2a7` 修改了主叠加层的 DPR 计算，是需要重点回归验证的历史变更；但不能仅凭提交历史确认唯一根因。

总览频谱在 `spectrum-interaction.ts` 中使用独立的 `getBoundingClientRect()` 与窗口帧范围逻辑处理滑动窗口，本次不应改变该分支。

## Affected files and components

| 文件/组件 | 影响类型 | 计划变更或验证 | 风险与检测方式 |
| --- | --- | --- | --- |
| `packages/ui/src/spectrum-interaction.ts` | 直接变更 | 统一主频谱鼠标坐标到逻辑绘图区的转换，并保证拖动时使用当前几何信息 | 可能改变边界、自动平移和 seek 时机；主频谱交互回归测试覆盖 |
| `packages/ui/src/spectrum-main-overlay.ts` | 直接变更 | 使红色播放线与主频谱交互使用相同的逻辑宽度/帧范围基准 | 可能影响播放头、悬停线及 DPR 绘制；覆盖 CSS 尺寸与 DPR 场景 |
| `packages/ui/src/spectrum-layout.ts` | 可能变更 | 若需要，提供可测试的主频谱逻辑坐标转换函数 | 共享 helper 误用于总览频谱；通过 API 范围和总览回归测试限制 |
| `packages/ui/test/index.mjs` | 测试变更 | 添加主频谱坐标换算及拖动回归测试，覆盖 CSS/内部像素尺寸、滚动后的 rect 和 DPR | 测试可能依赖 DOM；优先使用纯函数测试，必要时使用项目已有运行环境 |
| `docs/specs/main-spectrum-drag-offset.md` | 记录变更 | 记录实际实现、验证命令和提交哈希 | 仅文档影响 |

下列既有工作区变更不属于本 Spec：`app/vite.config.ts`、`app/certs/` 以及 `packages/core/models/*.onnx`，实施时不得纳入提交。

## Compatibility, migration, and operational impact

- 不涉及 API、持久化数据、数据库或迁移。
- 不改变总览频谱滑动窗口的数据契约和交互入口。
- 不新增依赖，不改变构建产物格式。
- 用户可见影响仅为主频谱红线与鼠标位置恢复一致；播放、缩放、主频谱边缘自动平移及总览拖动需保持现有兼容性。
- 页面滚动、窗口 resize、浏览器缩放和设备像素比场景依赖运行时的当前 DOM 几何信息，不能缓存过期的 Canvas 尺寸。

## Risks and mitigations

- **DPR/CSS 尺寸不一致**：Canvas 内部像素尺寸可能与 CSS 显示尺寸不同。使用同一逻辑 CSS 坐标系进行事件换算和绘制，并测试 DPR 非 1。
- **音频时间与频谱帧数不一致**：播放线由音频时间反算，拖动由帧数换算时间。测试非整数持续时间和首尾边界，必要时统一帧/时间转换。
- **主频谱自动平移改变 offset**：拖到边缘时当前实现会调整 `spectrumOffset`。测试自动平移前后红线位置和可见窗口，避免把该行为误改为总览窗口拖动。
- **总览回归**：总览逻辑与主频谱共享状态但不是同一坐标路径。保留其现有代码路径，并执行总览窗口拖动的回归测试。
- **既有脏工作区干扰**：提交前只暂存 Spec、批准的源码和测试文件，检查 staged diff，排除用户已有变更。

## Test strategy and execution order

1. **Red — 聚焦坐标测试**：先在 `packages/ui/test/index.mjs` 增加主频谱坐标映射测试，验证在 Canvas CSS 宽度与内部像素宽度不一致、Canvas 位于非零页面位置、DPR 非 1 时，事件位置映射到绘图逻辑坐标不产生右偏；预期新增测试失败。
2. **Green — 坐标链路修复**：实现最小生产代码修复，使主频谱交互和红色播放线共享同一逻辑宽度与帧/时间换算；运行聚焦测试至通过。
3. **回归 — 总览与公共布局**：运行 `pnpm --filter @ohm/ui test`，验证现有状态、时间线和控制器测试，并覆盖总览拖动不受影响的检查。
4. **类型与构建**：运行 `pnpm --filter @ohm/ui typecheck` 以及必要的 `pnpm --filter @ohm/ui build`；若构建会生成 `dist`，仅将批准的源码、测试和 Spec 纳入提交。
5. **全量回归**：运行 `pnpm typecheck`、`pnpm test`；若仓库全量测试耗时或受既有环境问题影响，记录实际结果和与本变更的关系。

Phase 3 期间未运行会写入构建产物的测试/构建命令，以遵守只读影响评估边界。

## Implementation plan

### Task 1 — 修复主频谱坐标一致性并添加回归测试

- 先添加最小可验证的主频谱坐标转换/拖动回归测试并确认 Red。
- 检查并统一主频谱事件坐标、Canvas CSS 逻辑坐标、DPR 和频谱帧/音频时间之间的换算基准。
- 仅在必要范围内修改 `spectrum-interaction.ts`、`spectrum-main-overlay.ts` 或 `spectrum-layout.ts`；不修改总览拖动实现。
- 运行聚焦测试、UI 包测试、类型检查和仓库回归测试。
- 更新本 Spec 的实际变更和验证证据，并为 Task 1 创建一个独立 Conventional Commit。

## Implementation status

Task 1 已实现，待创建对应 Git 提交。

## Actual changes

- 在 `packages/ui/src/spectrum-layout.ts` 增加 Canvas 逻辑尺寸解析，以及频谱帧与音频时间的互逆转换函数。
- 主频谱交互改用精确逻辑绘图区宽度，并使用未取整的帧值转换音频时间，避免事件位置与红线位置因取整或固定 `FRAME_SEC` 产生偏差。
- 主频谱叠加层改用渲染后的 CSS rect 计算逻辑宽高和 DPR，并使用与交互一致的音频时间到频谱帧转换；红色播放线不再额外取整绘制位置。
- `packages/ui/test/index.mjs` 增加 CSS/内部像素尺寸、非整数持续时间和主叠加层红线坐标回归测试。
- 未修改总览频谱滑动窗口的实现。

## Verification evidence

- Red：直接运行 `node --test packages/ui/test/index.mjs` 时，新增导出函数不存在，2 个新增基础测试失败；随后补充主叠加层测试并修正了测试中错误使用 `totalFrames - 1` 作为绘图分母的问题。
- Green：`pnpm --filter @ohm/ui test` — 6/6 通过。
- `pnpm --filter @ohm/ui typecheck` — 通过。
- `pnpm typecheck` — 4/4 workspace 包通过。
- `pnpm --filter @ohm/ui lint` — 0 warnings, 0 errors。
- `git diff --check` — 通过。
- `pnpm test`：core、runtime、ui 测试均通过；app 浏览器 worker 测试在初始化外部 Pyodide/推理 worker 时长时间无输出，已中断，因此全量测试没有完整通过结论。

## Deviations from plan

- 实现同时修复了帧/时间转换基准和 Canvas 精确 CSS 尺寸的取整差异，因为两者都会造成主频谱红线与鼠标位置不一致。
- 未新增或修改依赖、API、数据格式及总览频谱逻辑。

## Final impact assessment

- 预期效果：主频谱点击和拖动时，红色播放线与鼠标在相同逻辑绘图区位置；页面滚动、窗口尺寸变化、浏览器缩放和 DPR 非 1 时继续使用当前 DOM 几何信息。
- 已观察效果：坐标 helper、主叠加层绘制和 UI 回归测试均验证通过；仓库类型检查和 UI lint 通过。
- 未完成验证：`@ohm/app` 浏览器 worker 测试因外部 worker 初始化阻塞，未能获得结果。

## Task commit

待提交；提交后补录哈希。
