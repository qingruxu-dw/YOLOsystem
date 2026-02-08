@echo off
REM Performance Comparison: Fusion vs Simple Pipeline

echo ================================
echo Fusion vs Simple Pipeline Test
echo ================================
echo.

REM Check if dataset exists
if not exist "datasets\foggy_cityscapes" (
    echo ERROR: datasets\foggy_cityscapes not found
    echo Please download Foggy Cityscapes first
    exit /b 1
)

echo Starting performance comparison...
echo.

REM Test medium fog (beta=0.01)
echo [1/3] Testing medium fog (beta=0.01)...
python compare_fusion_vs_simple.py --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val --output-dir outputs\comparison_medium --pattern "*beta_0.01.png" --strategy adaptive --max-images 100

echo.
echo [2/3] Testing light fog (beta=0.005)...
python compare_fusion_vs_simple.py --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val --output-dir outputs\comparison_light --pattern "*beta_0.005.png" --strategy adaptive --max-images 100

echo.
echo [3/3] Testing dense fog (beta=0.02)...
python compare_fusion_vs_simple.py --input-dir datasets\foggy_cityscapes\leftImg8bit_foggy\val --output-dir outputs\comparison_dense --pattern "*beta_0.02.png" --strategy adaptive --max-images 100

echo.
echo ================================
echo Comparison completed!
echo ================================
echo.
echo Results saved to:
echo   - outputs\comparison_medium\
echo   - outputs\comparison_light\
echo   - outputs\comparison_dense\
echo.
echo View reports:
echo   - comparison_report.md
echo   - comparison_report.json
echo.

pause
