# 来源与封装范围

## 原方法

本地 `pianogen3/scripts/` 已验证实现的可移植子集。主要算法来自 `yanhua_v4_reconcile.py`、`yanhua_v4_dynamics.py`、`yanhua_natural.py`、`ophelia_v6_reading_score.py` 及新增视频处理入口。只复制实际调用的排谱、MIDI、合成和验证函数，移除历史歌曲缓存读取、绝对环境路径和按歌曲名触发的特殊分支。新的封装负责媒体准备、依赖解析、版本分配、阶段恢复和输出发布。

## 模型与依赖

- [TransKun 官方仓库](https://github.com/Yujia-Yan/Transkun)：软件包 2.0.1，使用自带 2.0 配置及默认未扩展踏板音长的权重。代码 MIT，许可证保存在 `assets/licenses/TransKun-MIT.txt`。
- [ByteDance 钢琴模型推理仓库](https://github.com/qiuqiangkong/piano_transcription_inference)：软件包 0.0.6；使用已验证的本地官方 CRNN 音符／踏板检查点；原发布文件位于 [Zenodo](https://zenodo.org/records/4034264)。代码及模型各自许可应以来源为准，不把代码许可自动扩展为所有资产许可。
- [PyTorch 历史版本安装说明](https://pytorch.org/get-started/previous-versions/)：固定 torch/torchaudio 2.5.1，CPU 或 CUDA 12.1 软件源。
- VexFlow 4.2.2、jsdom 25.0.0；依赖及原包许可证随 `assets/node_modules.tar.gz` 保存，版本锁定在 `package-lock.json`。SVG→PDF 使用 librsvg，再由 PyMuPDF 合并。

## 钢琴音色

15 个 WAV 从用户本项目 `tools/samples` 原样复制，封装内以 MIDI 音高命名。当前目录没有可核实的原始采样发布信息，故不虚构音色名称或授予额外分发许可。这里是用户现有工作区的可复现封装；若要对外发布，应先补齐音色来源或替换成来源清楚的采样。替换后更新校验清单并重新验证渲染。

## 权重校验

- `assets/models/bytedance.pth`：`c3fa9730725bf4a762f1c14bc80cd5986eacda01b026f5a4a2525cd607876141`
- `assets/models/transkun-2.0.pt`：`50a80010effc2a59ffcd068a95cd2b29bd7f23a27a3515bc3ccd209c89a3d44c`
- `assets/models/transkun-2.0.conf`：`d3d989214eb148230ee5df476d994dcde6af595904d3f968f1221d2e3bea5ac6`

模型、配置和音色的完整清单见 `assets/checksums.json`。部署时逐项校验，损坏时直接失败，不静默下载替代模型。
