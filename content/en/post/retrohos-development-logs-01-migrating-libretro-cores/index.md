---
title: "Retrohos Development Log #1: Migrating Libretro Cores"
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

> The following content is a rough translation by *Github Copilot* of the original post in Chinese.

The goal of Retrohos is to serve as the implementation of the RetroArch Frontend on OpenHarmony/HarmonyOS and to complete the porting of emulator cores from various communities, while also supporting the vision of distributed applications in the Harmony framework (e.g., experiences such as using a large screen display while the phone functions as a controller).

Since I previously only have rudimentary experience in Android mobile development (controlling Android screen casting via NDK), this log is merely a record of the application development exploration process, not the best implementation, and is for reference only.

## What is Libretro

RetroArch is a very popular retro gaming console solution. It doesn't provide any emulator functionality itself but acts as an aggregator for community emulators (Emulator Core) and provides a common interface definition, [Libretro](https://www.libretro.com/index.php/api). The following introduction is generated by ChatGPT:

> Libretro is a lightweight game console emulator core framework. Through the Libretro API, game cores (Core) can be separated from the frontend (Frontend), enabling cross-platform porting of cores and supporting multiple frontends such as RetroArch, Lakka, etc. The Libretro core is a dynamic link library that, through the Libretro API, can initialize game cores, handle input/output, and render audio and video.

Libretro effectively separates the components of the emulator, allowing us to ignore the implementation of existing frontends (and their different GUI technology stacks) and focus on porting the core and integrating it with the system. Therefore, the goal of the Retrohos project is to implement a frontend compatible with the Libretro API while porting community Emulator Cores, rather than a complete port of RetroArch.

## Preparing the NDK Environment

For OpenHarmony development, I recommend using the HarmonyOS commercial release and the developer beta SDK provided by Huawei, rather than the community open-source version of OpenHarmony. Although the two currently have little difference (HarmonyOS Next Beta1 is based on OpenHarmony 5.0.0.xx), the only available and mature ecosystem is HarmonyOS. This is an unavoidable reality.

Fortunately, OpenHarmony NDK development is similar to Android and does not require the official IDE (DevEco Studio). The following development content will be based on command-line tools and performed on a Linux platform (Ubuntu 22.04.3 LTS).

Taking the latest official Linux version SDK (`5.0.0.800`) as an example, let's assume the `commandline-tools.zip` archive has been downloaded and extracted to the `$HOME` directory. To differentiate version numbers, rename the extracted directory to `command-line-tools/5.0.0.800` and create an `env.sh` file with the following content:

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

Afterward, when using it, simply execute `source env.sh`. This file mainly serves the following purposes:
- Sets SDK-related environment variables. The variable names here are not specific requirements, just for easy reference.
- `$NDK_ROOT` points to the root directory of the NDK, making it easy to reference the compilation chain tools and link the `sysroot`.
- Prepend the paths of `cmake` and `llvm` to `$PATH`, overriding the system's default compilation toolchain to ensure the use of the NDK-provided toolchain.

According to the [official NDK development documentation](https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/ndk-development-overview-V5), when calling the NDK-provided `cmake` compiler, the following parameters need to be passed (using `aarch64-linux-ohos` as an example):

```bash
cmake -DOHOS_STL=c++_shared -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/ohos.toolchain.cmake .
```

Among them, `CMAKE_TOOLCHAIN_FILE` is the most important, specifying the configuration file of the compilation chain, which will automatically link the `sysroot` provided by the NDK.

## Porting Libretro Cores

### Attempting to Port `libretro-super`

The [`libretro/libretro-super`](https://github.com/libretro/libretro-super) repository is a project for managing Libretro cores and RetroArch compilation scripts on different platforms, organized according to the target triplet identifier. Initially, I thought that OpenHarmony's underlying system was just the Linux kernel's `aarch64` architecture, and the dynamic libraries in the toolchain were quite common, unrelated to the SDK. Therefore, I attempted to directly port this super project. Unfortunately, due to the fact that the `ohos` / `OHOS` identifiers are not supported by many upstream dependencies and it will not automatically downgrade to the aarch64 Linux build process, a brute-force port would only lead to endless dependency resolution.

Therefore, the most appropriate solution is to use only the Libretro Core build process (forking repositories one by one for modification 😓) and abandon the overall porting of `libretro-super`, which is too messy. At the start of this project, we focused on porting the ppsspp core: it provides the option to build the `ppsspp-libretro` dynamic library with few third-party dependencies and was the first core to be successfully built.

### Compiling the `ppsspp-libretro` Core

#### Building the `ppsspp-ffmpeg` Static Library

The [`hrydgard/ppsspp`](https://github.com/Retrohos/ppsspp) repository introduces multiple third-party dependencies through `.gitmodules`, among which only the `hrydgard/ppsspp-ffmpeg` maintained by itself requires additional handling to add the `OHOS` architecture string. Refer to `android_arm64_v8a.sh` and correspondingly add the content of the `ohos_arm64-v8a.sh` file as follows:

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

After preparing the NDK environment with `source env.sh`, execute the build script to generate static library files such as `libavcodec.a` under `ohos/arm64`, as well as header files under the `include` directory. After submitting the relevant changes, you can point the submodule in `ppsspp-libretro` to our modified repository [`Retrohos/ppsspp-ffmpeg`](https://github.com/Retrohos/ppsspp-ffmpeg.git).

#### Building `ppsspp`

The porting of `ppsspp-libretro` itself is relatively simple, only requiring the addition of OHOS architecture judgment in `CMakeLists.txt` and modification of the compilation options in `CMakeLists.txt`. The patch content is as follows:

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

After modifying, execute the following command to build the `ppsspp-libretro` core:
```bash
cmake -DOHOS_STL=c++_shared -DOHOS_ARCH=arm64-v8a -DOHOS_PLATFORM=OHOS -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/ohos.toolchain.cmake -DLIBRETRO=ON -DUSING_EGL=ON -DUSING_GLES2=ON . -Bbuild
```
which will generate the `build` directory, and execute `make -j4` within it to generate the `lib/ppsspp_libretro.so` file.

## Future Plans

As the project's initial stage, the completion of a core port is sufficient for subsequent frontend development. The next step is to design a simple Libretro frontend and bind it through ArkTS NAPI to quickly implement framebuffer output. As for more important GUI design, we can wait for the improvement of the DevEco Studio development suite (～o￣3￣)～