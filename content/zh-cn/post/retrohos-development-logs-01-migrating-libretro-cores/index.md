---
title: "Retrohos 开发日志其一: 移植 Libretro Cores"
description: 
slug: retrohos-development-logs-01-migrating-libretro-cores
date: 2024-09-07 22:14:46+0800
image: ppsspp-libretro-core.png
categories:
    - practice
tags:
    - RetroArch
    - OpenHarmony 
---

Retrohos 的目标是作为 OpenHarmony/HarmonyOS 上 RetroArch Frontend 的实现以及完成各社区 Emulator 核心的移植，并同时支持鸿蒙框架关于**分布式应用**的愿景（如大屏显示同时手机作为手柄的体验）。
由于本人之前只有粗浅的 Android 移动端开发经验（[通过NDK控制Android投屏](https://github.com/lasso-sustech/AndroidScreenCaster)），本日志仅作为应用开发探索过程中的记录，并非最佳实现，仅供参考。

## 什么是 Libretro

RetroArch 是一个非常流行的怀旧游戏主机解决方案。它本身并不提供任何仿真器的功能 ~~只是大自然的搬运工~~，只是提供了聚合社区仿真器（Emulator Core）的平台，以及一个通用的接入接口定义 [Libretro](https://www.libretro.com/index.php/api)。以下简介生成自 ChatGPT :

> Libretro 是一个轻量级的游戏机模拟器核心框架，通过 Libretro API，可以将游戏核心（Core）与前端（Frontend）分离，实现核心的跨平台移植，同时支持多种前端，如 RetroArch、Lakka 等。Libretro 核心是一个动态链接库，通过调用 Libretro API，可以实现游戏核心的初始化、输入输出、音频视频渲染等功能。

Libretro 很好地将模拟器的构成区分开来，使得我们可以忽略现有前端的实现（以及各异的GUI技术栈），专注于核心的移植以及与系统的对接。因此 Retrohos 项目的目标是将实现一个兼容 Libretro API 的前端，同时移植社区的 Emulator Core，而非 RetroArch 的完整移植。

## 准备 NDK 环境

目前关于 OpenHarmony 的开发，本人推荐直接使用 HarmonyOS 商业发行版以及华为提供的开发者内测SDK，而非社区开源的 OpenHarmony 版本；虽然两者目前差异度不大（HarmonyOS Next Beta1 基于 OpenHarmony 5.0.0.xx），但目前唯一可用且成熟的生态环境只有 HarmonyOS，这是一个不得不接受的现状。
但好在 OpenHarmony NDK 开发与 Android 类似，并不需要依赖官方提供的IDE（DevEco Studio）；以下的开发内容将以命令行工具为基础，在 Linux 平台（Ubuntu 22.04.3 LTS）上进行。

以当前最新的官方Linux版本SDK（`5.0.0.800`）为例，以下假设 `commandline-tools.zip` 压缩包已经下载并解压到 `$HOME` 目录下；为了方便区分版本号，将解压后的目录重新命名为 `command-line-tools/5.0.0.800`，并在其中创建 `env.sh` 文件，内容如下：

```bash
#!/bin/bash
export OHOS_ROOT=$HOME/command-line-tools/5.0.3.800
export SDK_ROOT=$OHOS_ROOT/sdk/default/openharmony

## export NDK Path
export NDK_ROOT=$SDK_ROOT/native
export PATH=$NDK_ROOT/build-tools/cmake/bin:$PATH
export PATH=$NDK_ROOT/llvm/bin:$PATH

## export OHOS command line tools
export PATH=$OHOS_ROOT/tool/node/bin:$PATH
export PATH=$OHOS_ROOT/bin:$PATH
```

之后在使用时，只需执行 `source env.sh` 即可。该文件主要起到以下作用：
- 设置SDK相关环境变量，这里变量名称没有特殊要求，只是为了方便后续引用；
- `$NDK_ROOT` 指向 NDK 的根目录，方便后续引用编译链工具，以及链接 `sysroot`；
- `$PATH` 中前置添加了 `cmake` 和 `llvm` 的路径，会覆盖掉系统默认的编译工具链， 以确保使用 NDK 提供的工具链。

另外根据[官方NDK开发文档](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ndk-development-overview-V5)，在调用 NDK 提供的 `cmake` 编译时，需要传入以下参数（以 `aarch64-linux-ohos` 为例）：

```bash
cmake -DOHOS_STL=c++_shared -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/ohos.toolchain.cmake .
```

其中 `CMAKE_TOOLCHAIN_FILE` 最为重要，指定了编译链的配置文件，该文件会自动链接 NDK 提供的 `sysroot`。

## 移植 Libretro 核心

### `libretro-super` 移植尝试

[`libretro/libretro-super`](https://github.com/libretro/libretro-super) 仓库是一个用于管理 Libretro 核心以及 RetroArch 在不同平台编译脚本的管理项目，其按照 *target triplet* 标识与组织不同平台下的构建流程。
最初，我以为 OpenHarmony 底层不过是 Linux kernel 的 `aarch64` 架构，工具链中的动态库也都很常见，与 SDK关系不大，因此尝试直接移植这个超级项目。然而可惜的是，由于 `ohos` / `OHOS` 标识并非众多上游依赖支持的平台，并且它也不会被自动降级到 aarch64 Linux 的构建流程中，暴力移植只会陷入到无限的依赖解决中。

因此最合适的解决方案，是仅采用 Libretro Core 的构建流程（逐个 Fork 仓库进行修改😓），抛弃 `libretro-super` 这个过于杂乱的项目的整体移植。在本项目的启动阶段，我们聚焦于 ppsspp 核心的移植：它本身提供了 `ppsspp-libretro` 的动态库构建选项，三方依赖较少，是第一个成功构建的核心。

### 编译 `ppsspp-libretro` 核心

#### `ppsspp-ffmpeg` 静态库构建

[`hrydgard/ppsspp`](https://github.com/Retrohos/ppsspp) 通过 `.gitmodules` 引入了多个三方依赖，其中只有其自己维护的 `hrydgard/ppsspp-ffmpeg` 需要额外处理 `OHOS` 架构字符串的添加，参考 `android_arm64_v8a.sh` 并对应添加 `ohos_arm64-v8a.sh` 文件内容如下：

```bash
#!/bin/bash

if [ "$NDK_ROOT" = "" ]; then
    echo "Please set NDK_ROOT to your OpenHarmony NDK location."
    exit 1
fi

NDK_PLATFORM="aarch64-linux-ohos"
NDK_PLATFORM_LIB=$NDK_ROOT/sysroot/usr/lib/$NDK_PLATFORM

NDK_PREBUILTLLVM=$NDK_ROOT/llvm

set -e

GENERAL="\
   --enable-cross-compile \
   --enable-pic \
   --cc=$NDK_PREBUILTLLVM/bin/clang \
   --cross-prefix=$NDK_PREBUILTLLVM/bin/aarch64-unknown-linux-ohos- \
   --ar=$NDK_PREBUILTLLVM/bin/llvm-ar \
   --ld=$NDK_PREBUILTLLVM/bin/clang \
   --nm=$NDK_PREBUILTLLVM/bin/llvm-nm \
   --ranlib=$NDK_PREBUILTLLVM/bin/llvm-ranlib"

MODULES="\
   --disable-avdevice \
   --disable-filters \
   --disable-programs \
   --disable-network \
   --disable-avfilter \
   --disable-postproc \
   --disable-encoders \
   --disable-protocols \
   --disable-hwaccels \
   --disable-doc"

VIDEO_DECODERS="\
   --enable-decoder=h264 \
   --enable-decoder=mpeg4 \
   --enable-decoder=mpeg2video \
   --enable-decoder=mjpeg \
   --enable-decoder=mjpegb"

AUDIO_DECODERS="\
    --enable-decoder=aac \
    --enable-decoder=aac_latm \
    --enable-decoder=atrac3 \
    --enable-decoder=atrac3p \
    --enable-decoder=mp3 \
    --enable-decoder=pcm_s16le \
    --enable-decoder=pcm_s8"

DEMUXERS="\
    --enable-demuxer=h264 \
    --enable-demuxer=m4v \
    --enable-demuxer=mpegvideo \
    --enable-demuxer=mpegps \
    --enable-demuxer=mp3 \
    --enable-demuxer=avi \
    --enable-demuxer=aac \
    --enable-demuxer=pmp \
    --enable-demuxer=oma \
    --enable-demuxer=pcm_s16le \
    --enable-demuxer=pcm_s8 \
    --enable-demuxer=wav"

VIDEO_ENCODERS="\
	  --enable-encoder=huffyuv
	  --enable-encoder=ffv1"

AUDIO_ENCODERS="\
	  --enable-encoder=pcm_s16le"

MUXERS="\
  	--enable-muxer=avi"

PARSERS="\
    --enable-parser=h264 \
    --enable-parser=mpeg4video \
    --enable-parser=mpegaudio \
    --enable-parser=mpegvideo \
    --enable-parser=aac \
    --enable-parser=aac_latm"

function build_arm64
{
    # no-missing-prototypes because of a compile error seemingly unique to aarch64.
./configure --logfile=conflog.txt --target-os=linux \
    --prefix=./ohos/arm64 \
    --arch=aarch64 \
    ${GENERAL} \
    --extra-cflags=" --target=aarch64-linux-ohos -no-canonical-prefixes -fdata-sections -ffunction-sections -fno-limit-debug-info -funwind-tables -fPIC -O2 -DOHOS -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/ohos.toolchain.cmake -Dipv6mr_interface=ipv6mr_ifindex -fasm -fno-short-enums -fno-strict-aliasing -Wno-missing-prototypes" \
    --disable-shared \
    --enable-static \
    --extra-ldflags=" -B$NDK_PREBUILTLLVM/bin/aarch64-unknown-linux-ohos- --target=aarch64-linux-ohos -Wl,--rpath-link,$NDK_PLATFORM_LIB -L$NDK_PLATFORM_LIB  -nostdlib -lc -lm -ldl" \
    --enable-zlib \
    --disable-everything \
    ${MODULES} \
    ${VIDEO_DECODERS} \
    ${AUDIO_DECODERS} \
    ${VIDEO_ENCODERS} \
    ${AUDIO_ENCODERS} \
    ${DEMUXERS} \
    ${MUXERS} \
    ${PARSERS}

make clean
make -j4 install
}

build_arm64
```

通过之前提到的 `source env.sh` 准备好 NDK 环境后，随后执行该构建脚本，在 `ohos/arm64` 下生成 `libavcodec.a` 等静态库文件，以及 `include` 目录下的头文件。
提交相关更改后，即可在 `ppsspp-libretro` 中将 submodule 指向我们修改后的仓库 [`Retrohos/ppsspp-ffmpeg`](https://github.com/Retrohos/ppsspp-ffmpeg.git).


#### `ppsspp` 构建

`ppsspp-libretro` 本身的移植较为简单，只需要在 `CMakeLists.txt` 中添加 OHOS 架构的判断，以及修改 `CMakeLists.txt` 中的编译选项，patch 内容如下：

```diff
diff --git a/CMakeLists.txt b/CMakeLists.txt
index ad2be076b..2fe1e619c 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -99,8 +99,12 @@ if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
 	set(ANDROID ON)
 endif()
 
+if(${CMAKE_SYSTEM_NAME} MATCHES "OHOS")
+	set(OHOS ON)
+endif()
+
 # We only support Vulkan on Unix, macOS (by MoltenVK), Android and Windows.
-if(ANDROID OR WIN32 OR (UNIX AND NOT ARM_NO_VULKAN))
+if(OHOS OR ANDROID OR WIN32 OR (UNIX AND NOT ARM_NO_VULKAN))
 	set(VULKAN ON)
 endif()
 
@@ -192,7 +196,7 @@ if(USE_CCACHE)
 	include(ccache)
 endif()
 
-if(UNIX AND NOT (APPLE OR ANDROID) AND VULKAN)
+if(UNIX AND NOT (APPLE OR ANDROID OR OHOS) AND VULKAN)
 	if(USING_X11_VULKAN)
 		message("Using X11 for Vulkan")
 		find_package(X11)
@@ -225,7 +229,7 @@ if(LIBRETRO)
 	endif()
 endif()
 
-if(ANDROID)
+if(ANDROID OR OHOS)
 	set(MOBILE_DEVICE ON)
 	set(USING_GLES2 ON)
 endif()
@@ -382,7 +386,7 @@ if(NOT MSVC)
 	# NEON optimizations in libpng17 seem to cause PNG load errors, see #14485.
 	add_definitions(-DPNG_ARM_NEON_OPT=0)
 
-	if(ANDROID)
+	if(ANDROID OR OHOS)
 		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")
 	endif()
 	if(CLANG)
@@ -474,7 +478,7 @@ if(NOT MSVC)
 		if(${CMAKE_SYSTEM_NAME} STREQUAL "NetBSD")
 			add_definitions(-D_NETBSD_SOURCE)
 		endif()
-	elseif(ANDROID)
+	elseif(ANDROID OR OHOS)
 		add_definitions(-fsigned-char)
 	endif()
 else()
@@ -943,6 +947,10 @@ if(USE_FFMPEG)
 				elseif(X86)
 					set(PLATFORM_ARCH "android/x86")
 				endif()
+			elseif(OHOS)
+				if(ARM64)
+					set(PLATFORM_ARCH "ohos/arm64")
+				endif()
 			elseif(IOS)
 				if(IOS_PLATFORM STREQUAL "TVOS")
 					set(PLATFORM_ARCH "tvos/arm64")
@@ -1200,7 +1208,7 @@ else()
 endif()
 
 # Arm platforms require at least libpng17.
-if(ANDROID OR ARMV7 OR ARM64 OR ARM OR IOS)
+if(OHOS OR ANDROID OR ARMV7 OR ARM64 OR ARM OR IOS)
 	set(PNG_REQUIRED_VERSION 1.7)
 else()
 	set(PNG_REQUIRED_VERSION 1.6)
```

修改完成后执行
```bash
cmake -DOHOS_STL=c++_shared -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/ohos.toolchain.cmake -DLIBRETRO=ON -DUSING_EGL=ON -DUSING_GLES2=ON . -Bbuild
```
创建 `build` 目录，并在其中执行 `make -j4` 即可生成 `lib/ppsspp_libretro.so` 文件。

## 后续计划

作为项目的启动阶段，一个核心的移植已经足够我们进行后续的前端开发。接下来计划先设计一个简单的 Libretro 前端，并通过 ArkTS NAPI 进行绑定调用，快速实现 framebuffer 的输出。至于更为重要的 GUI 设计，可以再等等 DevEco Studio 开发套件的完善(～o￣3￣)～
