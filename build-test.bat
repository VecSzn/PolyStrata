@echo off
call "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat" >nul 2>nul
set "WINSDK=C:/Program Files (x86)/Windows Kits/10"
set "SDKVER=10.0.26100.0"
set "WindowsSdkDir=%WINSDK%\"
set "WindowsSdkVersion=%SDKVER%"
set "WindowsSdkVerBinPath=%WINSDK%\bin\%SDKVER%\"
set "PATH=%WINSDK%\bin\%SDKVER%\x64;%PATH%"
set "INCLUDE=%INCLUDE%;%WINSDK%\Include\%SDKVER%\ucrt;%WINSDK%\Include\%SDKVER%\um;%WINSDK%\Include\%SDKVER%\shared;%WINSDK%\Include\%SDKVER%\winrt"
set "LIB=%LIB%;%WINSDK%\Lib\%SDKVER%\ucrt\x64;%WINSDK%\Lib\%SDKVER%\um\x64"
D:/llm/PolyStrata/.venv/Scripts/cmake.exe -S D:/llm/PolyStrataUpstream -B D:/llm/PolyStrataUpstream/build-up -DSTRATA_BUILD_TESTS=ON
if errorlevel 1 exit /b 1
D:/llm/PolyStrata/.venv/Scripts/cmake.exe --build D:/llm/PolyStrataUpstream/build-up --target poly_kv_q8_test -j 14
exit /b %errorlevel%
