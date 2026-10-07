# Piano Transcription Skill

把本地钢琴视频或音频转换为 **PDF 钢琴谱、MusicXML、MIDI 和采样钢琴重奏 MP3**，附原音对照、逐音筛选依据及版本记录。

流程：TransKun + ByteDance 双模型 → 原音频谱复核 → 可选视频音条复核 → 乐句力度与踏板处理 → 演奏文件与阅读谱分别导出。支持本地 GPU、CPU、Docker、失败续跑；旧版成品不会被覆盖。

## 一键部署

首次使用建议从本仓库 **Releases** 下载完整 `piano-transcription-skill.tar.gz`，解压后运行。完整包已包含两个模型、音色和排谱模块。

```bash
cd piano-transcription
./deploy.sh --docker --cpu
./run-docker.sh /path/to/video.mp4 /path/to/result --device cpu
```

GPU 容器需要宿主机 NVIDIA 驱动及 Container Toolkit：

```bash
./deploy.sh --docker
PIANO_DOCKER_GPU=1 ./run-docker.sh /path/to/video.mp4 /path/to/result --device cuda
```

已有 Python 3.11、Node ≥18、FFmpeg/ffprobe、rsvg-convert 等系统依赖时，也可以原生安装：

```bash
./deploy.sh --fresh
./run.sh /path/to/video.mp4 --output /path/to/result
```

从 Git 克隆时，模型权重由 `deploy.sh` 从固定版本 Release 下载并校验 SHA-256。私有仓库需要先通过 `gh auth login --web` 登录；也可在浏览器登录后下载完整部署包。文件名和目录名可含中文、空格。

## 常用参数

```bash
./run.sh /path/to/media-directory --output /path/to/result
./run.sh /path/to/Perfect.mp4 --output /path/to/result --meter 12/8 --bpm 100
./run.sh /path/to/video.mp4 --output /path/to/result --resume
./deploy.sh --check
```

`--bpm` 始终以四分音符为单位；未指定拍号时暂按 4/4。目录模式中的参数应用于所有曲目，不同拍号请分开调用。

输出目录：`result/歌曲名/原音对照版_vN/`。重奏 MP3 与提取的原音轨分开保存，另附 6 组试听片段及检查报告。原视频、个人运行缓存、虚拟环境及本机登录配置不在 Git 仓库内。

## 作为 Codex skill 使用

将仓库或完整部署包放到 `~/.codex/skills/piano-transcription/`，重新启动会话后用 `$piano-transcription` 调用。入口及约束见 [SKILL.md](SKILL.md)。

## 方法、验证与范围

- [部署说明](references/deployment.md)：新机器安装、容器、GPU、缓存及错误恢复。
- [算法方法](references/method.md)：时间轴、双模型筛选、重复音、视频几何和记谱。
- [验证记录](references/validation.md)：本机 L40、全新 CPU 环境、独立 Docker CPU 真实媒体测试及故障续跑。
- [来源与第三方资产](references/provenance.md)：模型、音色、依赖和校验清单。

适用于以钢琴为主的音视频，不包含人声／乐队混音分离。程序验证音符导出一致性、时值、连续性及文件有效性；谱面与音符准确性仍需演奏者复核，不等同于出版级人工扒谱。第三方资产的许可分别以来源为准。
