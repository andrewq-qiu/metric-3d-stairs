python infer.py \
    --cfg 'Metric3D/mono/configs/HourglassDecoder/vit.raft5.small.py' \
    --weights 'Metric3D/weight/metric_depth_vit_small_800k.pth' \
    --data 'data/test_stairs/test_stairs_annotations.json' \
    --out 'out'