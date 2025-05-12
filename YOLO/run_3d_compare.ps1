# Run the 3D comparison script with urban_tree_33 depth maps
python yolo_3d_compare.py `
    --image "train\images\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.jpg" `
    --label "train\labels\urban_tree_33_jpg.rf.82a6b61f057221ed1b39cd80344f5dab.txt" `
    --depth "..\outputs\urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab\depth_map.png" `
    --depth_masked "..\outputs\urban_tree_33_jpg_rf_82a6b61f057221ed1b39cd80344f5dab\depth_masked.png"
