# OhMyMelody

OhMyMelody 是一个基于 ONNX Runtime Web 的旋律提取网站。它直接在浏览器中运行旋律推理，目标是提供一个轻量、交互式的音频分析体验。

英文版请查看 [README.en.md](README.en.md)。

## 功能

- 基于浏览器的旋律提取
- 使用 ONNX Runtime Web 进行推理
- 支持音频分析的渐进式可视化
- 支持模型切换与分析流程管理

## 模型来源

本项目参考并借鉴了以下模型相关工作：

- Mamba 模型：https://github.com/Fan2me/Melody
- MFTFA：https://github.com/SmoothJing/MF-TFA_SD-MS
- MSNet：https://github.com/bill317996/Melody-extraction-with-melodic-segnet/

以上仓库仅作为技术参考与致谢。请务必查看原仓库对应的许可证、使用限制和署名要求。

## 特别致谢

特别感谢上述项目的作者和维护者，感谢他们在旋律提取、模型实现和开源贡献方面所做的工作。没有这些社区成果，就不会有这个项目的实现基础。

## 许可证

本仓库中的代码遵循根目录的 MIT License。

仓库中可能包含第三方模型权重或资源文件，这些内容受其原始许可证约束。重新分发或二次使用前，请先查看上面列出的原始来源。

## 运行

安装依赖并启动开发服务器：

1. pnpm install
2. pnpm dev

后续如果需要，也可以继续补充构建和部署说明。
