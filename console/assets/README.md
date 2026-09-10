# assets/

Wide-strip sample frames for the demo. The rest of the project ships no
image wider than 600 px, so `DefectDetector.predict_tiled` has never had
anything to tile; each file here is a 2048x1000 grayscale line-scan frame,
which is real strip geometry and 3.2x the 640 px the detector deploys at.

**Source:** https://huggingface.co/datasets/imaadd05/gc10-det
**Licence:** CC BY 4.0 -- attribution required, given below.
**Cite:** Lv, X.; Duan, F.; Jiang, J.J.; Fu, X.; Gan, L. Deep Metallic Surface Defect Detection: The New Benchmark and Detection Network. Sensors 2020, 20(6), 1562.

Unmodified frames, copied by `src/prepare_gc10.py`. `split` is the split
the frame sits in under `data/gc10-det`, so a model trained on that split
has not seen these images.

| file | what it shows | GC10 frame | split | labelled boxes | widest box |
|---|---|---|---|---|---|
| `strip_sample_rollmark.jpg` | roll mark (GC10 8_yahen rolled pit / 9_zhehen crease) | `img_03_4402117000_00003` | val | rolled_pit x4 | 200 px |
| `strip_sample_edge_defect.jpg` | edge defect, PROXY for edge crack (GC10 3_yueyawan crescent gap / 10_yaozhed waist fold) | `img_01_425006200_01173` | test | crescent_gap x1 | 308 px |
| `strip_sample_weldline.jpg` | coil-join weld line running across the strip (GC10 2_hanfeng) | `img_07_3403337700_00899` | test | weld_line x1 | 2006 px |

`strip_sample_edge_defect.jpg` is an **edge defect, not an edge crack**.
No public dataset has an edge-crack class; crescent gap and waist folding
are the closest honest proxies. See `reports/gc10_dataset.md`.
