# 两张BMP的白框反光传统算法试验

后续：用户提出固定白框结构后，已追加[固定反光场建模扣除](glare_fixed_field_20260912.md)，确认可显著削弱白框；下文仅描述最初两种通用算法的结果，不应作为否定固定模板校正的依据。

输入：`dataset/Image_20260912195524733.bmp`、`dataset/Image_20260912195528653.bmp`，均4024×3036 uint8 BGR。只做离线诊断，原图SHA256未变。

结论：可以压低高亮和增强局部刻字，不能据此宣称去除了反射或恢复真实缺陷纹理。

- OpenCV双边滤波估计低频亮度场，限制增益的局部亮度均衡：白亮区变暗，但矩形区域和圆形暗区仍在，部分立体明暗也被压平。
- LAB亮度通道CLAHE：刻字/细纹更明显，但反光仍在，纹理及噪声也会增强。
- 两图全通道255分别588589/588545px，均约4.82%；灰度>=250约5.45%/5.49%。这些是剪切候选，不是反光分割或缺陷标签。
- 未配准直接像素比较灰度相关.99928，共同全白占全图4.67668%，占第一张全白约97.07%。第一张全白但第二张灰度<250仅全图.00580%，反向.00740%。两图缺少明显互补曝光信息；噪声/小位移处差异不当作恢复来源。
- 全白位置在保存图中没有可用于恢复真实表面的亮度差。把255变成灰色并不恢复信息。[OpenCV inpaint](https://docs.opencv.org/4.9.0/d7/d8b/group__photo__inpaint.html)依靠周围内容补区域，不能作为真实裂纹、划伤、刻字证据，故未实施修补。

结果：`artifacts/bmw_glare_trial_20260912/outputs/index.html`，包含两图全尺寸原图PNG/亮度均衡/CLAHE/剪切标记、概览、刻字局部对比、report.json与verification.json。确切算法参数和源图/结果SHA均在report.json；可复现脚本`artifacts/bmw_glare_trial_20260912/run_trial.py`，输出目录必须不存在。

采集侧更有希望的途径是改变照明几何/扩散照明，避免光源形状直接映入镜面金属；偏振需在实际角度和材料上试验，并考虑短曝光下的光量损失。参考[漫射穹顶照明](https://www.edmundoptics.com/knowledge-center/application-notes/illumination/brightfield-illumination-with-dome-lights-for-machine-vision/)与[照明/偏振说明](https://www.edmundoptics.com/knowledge-center/application-notes/illumination/choose-the-correct-illumination/)。未识别实际光源型号；圆形暗区来源未确认，未将其擅自擦除。

若主要目标是OCR，可验证CLAHE区域预处理；若是表面缺陷/轮廓检测，需要重新验证算法输出，不能直接把本试验图作为生产输入。没有相机、照明硬件或生产检测验收。
