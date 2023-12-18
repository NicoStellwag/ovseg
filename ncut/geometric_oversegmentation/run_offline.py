import felzenszwalb_cpp
import open3d as o3d
from pathlib import Path
import numpy as np
import os.path
import json

ds_dir = "/mnt/hdd/scannet"
mesh_pattern = "*_vh_clean_2.ply"
mesh_files = [str(i) for i in Path(ds_dir).rglob(mesh_pattern)]

save_dir = "/mnt/hdd/scannet_segments"

kthr = 0.005 # threshold
segment_min_vert_num = 50 # not set in hydra config so assume default of class constructor

for mesh_file in mesh_files:
    scene_mesh = o3d.io.read_triangle_mesh(mesh_file)
    vertices = np.array(scene_mesh.vertices).astype(np.single)
    colors = np.array(scene_mesh.vertex_colors).astype(np.single)
    faces = np.array(scene_mesh.triangles).astype(np.intc)

    seg_indices, seg_connectivity = felzenszwalb_cpp.segment_mesh(vertices, faces, colors, kthr, segment_min_vert_num)

    scene_id = "/" + os.path.splitext(os.path.basename(mesh_file))[0]
    content = {
        "params": {
            "kThresh": kthr,
            "segMinVerts": segment_min_vert_num,
        },
        "sceneId": scene_id,
        "segIndices": seg_indices.tolist(),
        "segConnectivity": seg_connectivity.tolist()
    }
    
    filename = str(os.path.join(save_dir, scene_id[1:] + "." + f"{kthr:.6f}" + ".segs.json"))
    with open(filename, 'w') as f:
        f.write(json.dumps(content))
    print("Saved: ", filename)