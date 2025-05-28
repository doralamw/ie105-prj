FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    ANDROID_SDK_ROOT=/opt/android-sdk \
    PATH=$PATH:/opt/android-sdk/cmdline-tools/latest/bin:/opt/android-sdk/platform-tools:/opt/android-sdk/emulator
ENV ADB_SERVER_PORT=5037
RUN apt-get update && apt-get install -y --no-install-recommends \
      wget unzip openjdk-11-jdk-headless \
      adb strace \
      libx11-6 libxrender1 libxcb1 libxrandr2 libxtst6 libxi6 libxcomposite1 \
      libxcursor1 libgl1-mesa-dev libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $ANDROID_SDK_ROOT/cmdline-tools && cd $ANDROID_SDK_ROOT && \
    wget https://dl.google.com/android/repository/commandlinetools-linux-8512546_latest.zip \
      -O cmdline.zip && unzip cmdline.zip -d cmdline-tools && \
    mv cmdline-tools/cmdline-tools cmdline-tools/latest && rm cmdline.zip

RUN yes | sdkmanager --sdk_root=$ANDROID_SDK_ROOT --licenses && \
    sdkmanager --sdk_root=$ANDROID_SDK_ROOT \
      "platform-tools" \
      "emulator" \
      "platforms;android-29" \
      "system-images;android-29;default;x86_64"

RUN echo "no" | avdmanager create avd \
      --name test \
      --package "system-images;android-29;default;x86_64" \
      --device "pixel"

RUN mkdir -p /data/logs

# Thêm -read-only
ENTRYPOINT ["bash", "-c", "adb start-server && emulator -avd test -read-only -no-window -no-audio -gpu swiftshader_indirect -accel off -no-boot-anim -no-snapshot-load -no-snapshot-save -no-metrics -port 5554 && sleep infinity"]

