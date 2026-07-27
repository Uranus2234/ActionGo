# SteadyTray AprilTag 打印页

这些 PDF 用 AprilRobotics 官方 `tagStandard41h12` 预生成 tag 图生成，适配当前 `generic_sensor_policy_player --object-source apriltag` 的默认 family。

## 推荐先打印哪张

- 首次真机调试：`tagStandard41h12_id0_tagsize70mm_A4.pdf`
- 瓶子/小物体贴不下：`tagStandard41h12_id0_tagsize60mm_A4.pdf` 或 `50mm`
- 更远距离/更稳：`80mm`，但物理尺寸较大

> 注意：对 `tagStandard41h12`，`--tag-size` 是检测/位姿估计边长，即 5 个模块；PDF 中可见的 9x9 tag 图案总宽度为 `tag_size * 9/5`。

| PDF | 运行参数 | 可见 9x9 tag 宽度 |
|---|---:|---:|
| `tagStandard41h12_id0_tagsize50mm_A4.pdf` | `--tag-id 0 --tag-size 0.050` | 90 mm |
| `tagStandard41h12_id0_tagsize60mm_A4.pdf` | `--tag-id 0 --tag-size 0.060` | 108 mm |
| `tagStandard41h12_id0_tagsize70mm_A4.pdf` | `--tag-id 0 --tag-size 0.070` | 126 mm |
| `tagStandard41h12_id0_tagsize80mm_A4.pdf` | `--tag-id 0 --tag-size 0.080` | 144 mm |
| `tagStandard41h12_ids0-3_tagsize40mm_A4.pdf` | 单个 tag 用对应 `--tag-id N --tag-size 0.040` | 每个 72 mm |

## 打印要求

1. A4，100% / Actual size 打印。
2. 不要使用“适合页面 / Fit to page”。
3. 打印后用 PDF 上的 scale check 线或 100 mm ruler 检查是否被缩放。
4. 裁剪时保留灰色虚线内的白边，至少保留一圈白色 quiet zone。
5. 不要把 tag 贴在弯曲面上；建议贴在硬纸片/塑料片上，再固定到物体。

## action_go 检测测试

```bash
cd /path/to/action_go
build/generic_sensor_policy_player --spec actions/steadytray/config/generic_steadytray.yaml --object-source apriltag \
  --tag-id 0 --tag-size 0.070 --require-tag --camera-check
```

如果打印机有缩放误差，以实际测量的 `tag_size` 为准；对这些 PDF，`tag_size = 可见 9x9 tag 总宽度 * 5/9`。

## 小尺寸版本（整张可见黑白码边长 4–6 cm）

如果你说的“边长”是整张可见 9x9 黑白码的外边长，用这些 PDF：

| PDF | 可见黑白码边长 | action_go 参数 |
|---|---:|---:|
| `tagStandard41h12_id0_visible40mm_A4.pdf` | 40 mm | `--tag-id 0 --tag-size 0.0222` |
| `tagStandard41h12_id0_visible50mm_A4.pdf` | 50 mm | `--tag-id 0 --tag-size 0.0278` |
| `tagStandard41h12_id0_visible60mm_A4.pdf` | 60 mm | `--tag-id 0 --tag-size 0.0333` |
| `tagStandard41h12_id0_visible40-60mm_A4.pdf` | 一页含 40/50/60 mm | 按页面标签填写 |

说明：这些 PDF 的 `visibleXXmm` 指整张 9x9 图案的外边长；当前 AprilRobotics `tagStandard41h12` 的位姿 `tag_size` 对应 5 个模块，所以 `tag_size = visible_edge * 5 / 9`。打印后以实际测量值为准。

## 紧凑多 ID 版本（减少空白）

按用户反馈新增了紧凑排版，并提供反色视觉版本：

| PDF | 内容 | action_go 参数 |
|---|---|---|
| `tagStandard41h12_ids0-11_visible50mm_compact_inverted_A4.pdf` | 一页 12 个 ID，外边长 50 mm，反色视觉 | `--tag-size 0.0278`，`--tag-id` 按标签填 |
| `tagStandard41h12_ids0-5_visible60mm_compact_inverted_A4.pdf` | 一页 6 个 ID，外边长 60 mm，反色视觉 | `--tag-size 0.0333`，`--tag-id` 按标签填 |
| `tagStandard41h12_ids0-15_visible40mm_compact_inverted_A4.pdf` | 一页 16 个 ID，外边长 40 mm，反色视觉 | `--tag-size 0.0222`，`--tag-id` 按标签填 |

同时保留了 official polarity fallback：

- `tagStandard41h12_ids0-11_visible50mm_compact_official_A4.pdf`
- `tagStandard41h12_ids0-5_visible60mm_compact_official_A4.pdf`

如果反色版本在某个 detector 上识别不稳定，用 official 版本排查。

## 颜色直接黑白对调版（COLOR_SWAPPED）

这几个是按用户要求重新生成的版本：相对之前 PDF **黑白直接对调**，灰色虚线里面的第一圈视觉边框为黑色。

| PDF | 内容 | action_go 参数 |
|---|---|---|
| `tagStandard41h12_ids0-11_visible50mm_COLOR_SWAPPED_A4.pdf` | 一页 12 个 ID，外边长 50 mm | `--tag-size 0.0278`，`--tag-id` 按剪下的标签填 |
| `tagStandard41h12_ids0-5_visible60mm_COLOR_SWAPPED_A4.pdf` | 一页 6 个 ID，外边长 60 mm | `--tag-size 0.0333`，`--tag-id` 按剪下的标签填 |
| `tagStandard41h12_ids0-15_visible40mm_COLOR_SWAPPED_A4.pdf` | 一页 16 个 ID，外边长 40 mm | `--tag-size 0.0222`，`--tag-id` 按剪下的标签填 |
