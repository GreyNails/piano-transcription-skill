# 部署与运行

## GitHub 分发

Git 仓库保存代码、校验清单、音色与排谱依赖；两个较大的模型权重保存在固定版本 Release 的完整部署包中。`deploy.sh` 在缺少权重时自动读取 `assets/release.json`，从 Release 补齐并核对 SHA-256。私有仓库先用 `gh auth login --web` 登录，或在浏览器登录后下载完整包。已有完整包的部署不需要再次下载。

```bash
python3 scripts/fetch_assets.py --bundle /path/to/piano-transcription-skill.tar.gz
```

以上命令也可从本地完整包恢复缺失权重。`deploy.sh --check` 只检查，不触发下载。

## 当前机器：一条命令

```bash
cd /storage/homes/hongzj/dell/pianogen3/skill
./deploy.sh
```

默认优先复用同级项目已验证的 Python、FFmpeg、Node 和排谱依赖；不会修改复用环境中的软件包。随后检查 Python 导入、CUDA、两模型及音色的 SHA-256、FFmpeg/ffprobe、Node 排谱模块、SVG→PDF 工具。成功才写 `.deployment.json`。

处理一个文件、目录及失败续跑：

```bash
./run.sh ../add/vivalavida2.mp4 --output ../result --meter 4/4 --bpm 138
./run.sh ../add --output ../result
./run.sh ../add --output ../result --resume
./deploy.sh --check
```

目录模式逐首处理，参数适用于目录内所有曲目；不同拍号应分开调用。默认只枚举目录第一层的常见音视频格式，不解压、不递归、不下载媒体。文件名可含中文和空格。每次新运行建立独立 `原音对照版_vN`；同参数 `--resume` 复用中间结果。已有成品在新运行前后核对 SHA-256。

## 新机器：完整容器部署

支持目标为 Linux x86_64、Docker。首次构建需要能访问容器仓库、Debian、PyPI 和 PyTorch；模型、钢琴音色和锁定版本的 Node 排谱模块已包含，不依赖原项目目录，不需要重新下载权重或访问 npm。

```bash
# GPU 版镜像：安装 CUDA 12.1 版 PyTorch
./deploy.sh --docker
PIANO_DOCKER_GPU=1 ./run-docker.sh /path/to/video.mp4 /path/to/result --device cuda

# 没有 GPU 的机器
./deploy.sh --docker --cpu
./run-docker.sh /path/to/audio.wav /path/to/result --device cpu
```

GPU 容器还要求宿主机 NVIDIA 驱动与 NVIDIA Container Toolkit 已配置；脚本不会安装或更换驱动。默认容器运行不请求 GPU，只有 `PIANO_DOCKER_GPU=1` 时传入 `--gpus all`。CPU/GPU 构建共用本地镜像标签，重新构建会更新该标签。

容器只读挂载所选输入，输出和缓存落在指定输出目录中，以当前用户 UID/GID 写入。`--visual-config` 在容器中使用已挂载目录下的路径，例如把配置文件放入输入目录，传 `/input/calibration.json`。

## 新机器：原生 Python 环境

前提：Python 3.11、Node ≥18、FFmpeg/ffprobe（支持 libmp3lame、loudnorm、alimiter）、rsvg-convert 和字体可用。Debian/Ubuntu 上系统依赖可由管理员安装 `ffmpeg librsvg2-bin fonts-dejavu-core build-essential`，Node 使用满足版本要求的发行包。排谱模块从校验过的本地压缩包解压，不要求 npm 在线安装。

```bash
./deploy.sh --fresh          # skill/.venv 内安装 PyTorch 2.5.1 / CUDA 12.1 与 Python 依赖
./deploy.sh --fresh --cpu    # CPU 版；不改全局 Python
```

`PYTHON_BOOTSTRAP=/path/to/python3.11 ./deploy.sh --fresh` 可选择创建虚拟环境的 Python。`./deploy.sh --python /path/to/existing/python` 仅校验、复用指定环境，不向其中安装包。

`PIANO_FFMPEG`、`PIANO_FFPROBE`、`PIANO_NODE` 可指定工具位置。部署配置包含本机解析出的绝对路径；迁移后重新运行 `deploy.sh`，不要复制旧 `.deployment.json`。

Python 的直接依赖和传递依赖固定在 `requirements.txt`、`requirements.lock`，Node 包固定在 `package-lock.json` 及已打包模块。再次执行普通 `deploy.sh` 只复核已选环境，不把 GPU 配置切换为测试用 CPU 环境。

安装失败不会生成新的“成功”配置；已有有效配置保留。日志中的错误要先修复。网络问题检查实际下载域名、代理或软件源，不用关闭证书校验，也不替换成未知模型。CUDA 显存不足可减小并发（本入口已经顺序推理）或显式选择 `--device cpu`，保留失败日志后 `--resume`。

## 文件与缓存

```text
result/歌曲名/原音对照版_vN/
  歌曲名.pdf / 歌曲名.musicxml
  歌曲名.mid / 歌曲名.mp3
  歌曲名_原音频.mp3
  原音_钢琴重奏试听对比/
  最终检查.json / 运行参数.json / 版本说明.md
  逐音筛选依据.csv / *.json
skill/.runs/歌曲名_运行ID/
  config.json / pipeline.log / published.json
  work/歌曲名/{source.wav,transkun/,bytedance/,...}
```

用 `--work-dir /large-disk/piano-cache` 改缓存目录。当前推理按模型窗口计算，但频谱核对、音符比较和渲染仍读取整曲；长录音消耗更多 RAM，建议先按完整曲目边界分段。不要硬切连奏音符后直接拼 MIDI。实际测试范围见 `validation.md`。

## 迁移打包与 Codex 安装

```bash
python3 scripts/package.py
# 在 skill 的上一级生成 piano-transcription-skill.tar.gz 及 SHA-256 文件
```

压缩包包含 `SKILL.md`、代码、模型、音色、依赖版本和文档，排除本机虚拟环境、绝对路径部署配置、运行缓存和测试媒体。解压目录名为 `piano-transcription`。新机器解压后执行 `deploy.sh --docker`，或在系统依赖已安装时执行 `deploy.sh --fresh`。

需要让 Codex 自动发现时，将解压后的整个 `piano-transcription` 目录放入 `~/.codex/skills/`，然后重新启动会话。用户本次指定目录中已保存完整 skill；本任务不会擅自修改全局技能目录。
