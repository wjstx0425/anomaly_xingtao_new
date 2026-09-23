# 曝光测试原图偏黄排查（2026-09-11）

用户指定 `results/bmw_exposure/20260911_161054_385521/raw/front`。已检查原图、会话清单和代码，8张原图SHA256均与清单一致；未修改采集参数或原图。

- 本次300/400 μs，回读一致，gain12。front DA9805574型号MV-CU120-10UC为彩色；其余三台MV-CU120-10UM的PNG三通道逐像素相同。front视觉偏黄绿色，其他视图不能用于判断光源色偏。
- 偏色存在于融合前PNG，不是HDR处理造成。400μs全图B/G/R均值24.88/28.16/27.20；统一灰度30..180的像素均值85.82/96.05/95.31，B/G约.893。
- 前一次20260911_155941_921693同为400μs/gain12，视觉更中性，同口径B/G/R=92.63/94.54/95.75，B/G约.980。不同图选中像素不一定完全相同，该统计不是灰卡标定，不证明光照一致；但不能直接把本次偏黄归因于曝光/增益数值。
- 对比图：`artifacts/bmw_exposure_color_review_20260911/sessions_400us.jpg`；四视图：`four_views.jpg`。

代码初始化没有设置/读取白平衡、通道比例、光源预设或Gamma，session也没有实际PixelType。无法由PNG区分光源光谱和相机颜色设置的贡献。

另有确定的条件性转换缺陷：PFNC BayerRG8（RGGB）使用了OpenCV `COLOR_BAYER_RG2BGR`（实际BGGR），应使用 `COLOR_BAYER_RGGB2BGR`。独立合成验证RGGB输入R200/G100/B20时当前输出BGR[200,100,20]，正确为[20,100,200]。依据[OpenCV官方定义](https://docs.opencv.org/4.13.0/d8/d01/group__imgproc__color__conversions.html)。本次实际走Bayer还是BGR8_Packed分支未知，不能断定偏黄全由该缺陷引起；红蓝互换本身不会把中性灰变黄。此次原因排查未修改转换代码。

下一步：同一光源、400μs/gain12下对比MVS与程序输出，记录PixelFormat、白平衡模式和RGB比例/Gamma。MVS同样偏黄时检查照明/颜色设置；仅程序偏色时检查像素转换，注意MVS自身显示处理也可能不同。在实际照明下使用未过曝中性灰卡标定白平衡，若支持则一次标定后固定比例；不能用金属反光或黑背景作可靠白平衡基准。[白平衡功能说明](https://docs.baslerweb.com/balance-white)。

尚未读取现场相机当前配置或取得传感器Bayer原始帧，没有现场颜色标定或修复验收。
