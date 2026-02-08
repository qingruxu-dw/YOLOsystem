@echo off
REM Foggy Cityscapes 快速测试脚本 (Windows版)

echo ================================
echo Foggy Cityscapes 批量测试
echo ================================

REM 检查数据集是否存在
if not exist "datasets\foggy_cityscapes" (
    echo 错误: 找不到 datasets\foggy_cityscapes 目录
    echo.
    echo 请先下载 Foggy Cityscapes 数据集:
    echo 1. 访问 https://www.cityscapes-dataset.com/
    echo 2. 注册账号并登录
    echo 3. 下载 leftImg8bit_trainvaltest_foggy.zip
    echo 4. 解压到 datasets\foggy_cityscapes\
    exit /b 1
)

echo.
echo 开始测试验证集...
echo.

REM 测试中雾 (beta=0.01) - 推荐
echo [1/3] 测试中雾 (beta=0.01)...
python batch_test_fusion.py ^
    --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val ^
    --output-dir outputs\foggy_cityscapes_medium ^
    --pattern "*beta_0.01.png" ^
    --strategy adaptive

REM 测试轻雾 (beta=0.005)
echo.
echo [2/3] 测试轻雾 (beta=0.005)...
python batch_test_fusion.py ^
    --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val ^
    --output-dir outputs\foggy_cityscapes_light ^
    --pattern "*beta_0.005.png" ^
    --strategy adaptive

REM 测试浓雾 (beta=0.02)
echo.
echo [3/3] 测试浓雾 (beta=0.02)...
python batch_test_fusion.py ^
    --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val ^
    --output-dir outputs\foggy_cityscapes_dense ^
    --pattern "*beta_0.02.png" ^
    --strategy adaptive

echo.
echo ================================
echo 测试完成！
echo ================================
echo.
echo 结果目录:
echo   - outputs\foggy_cityscapes_medium\ (中雾)
echo   - outputs\foggy_cityscapes_light\  (轻雾)
echo   - outputs\foggy_cityscapes_dense\  (浓雾)
echo.
echo 查看汇总报告: type outputs\foggy_cityscapes_medium\summary.json
echo.

pause
