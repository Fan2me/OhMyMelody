# OhMyMelody

OhMyMelody is a web application for melody extraction powered by ONNX Runtime Web. It runs melody inference directly in the browser and is designed to provide a lightweight, interactive experience for audio analysis.

中文版请查看 [README.md](README.md)。

## Features

- Browser-based melody extraction
- ONNX Runtime Web inference
- Progressive visualization for audio analysis
- Model switching and analysis pipeline management

## Model Sources

This project references and adapts the following model-related work:

- Mamba models: https://github.com/Fan2me/Melody
- MFTFA: https://github.com/SmoothJing/MF-TFA_SD-MS
- MSNet: https://github.com/bill317996/Melody-extraction-with-melodic-segnet/

These repositories are listed here as technical references and acknowledgements. Please check the original repositories for their respective licenses, usage restrictions, and attribution requirements.

## Acknowledgements

Special thanks to the authors and maintainers of the projects above for their research, implementations, and open-source contributions. This project would not be possible without the community's work on melody extraction and browser-side machine learning.

## License

The code in this repository is distributed under the MIT License.

This repository may include third-party model weights or assets that are governed by their original licenses. Refer to the model sources above for details before redistributing or reusing them.

## Running

Install dependencies and start the development server:

1. pnpm install
2. pnpm dev

More deployment and build details can be added here later if needed.