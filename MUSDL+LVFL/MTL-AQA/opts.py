# directory containing frames
frames_dir = './data/frames'


# directory containing labels and annotations
info_dir = './data/info'
# i3d model pretrained on Kinetics, https://github.com/yaohungt/Gated-Spatio-Temporal-Energy-Graph
i3d_pretrained_path = './data/rgb_i3d_pretrained.pt'

# num of frames in a single video
num_frames = 103

# beginning frames of the 10 segments
segment_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 87]

# input data dims;
C, H, W = 3,224,224
# image resizing dims;
input_resize = 455, 256

# statistics of dataset
label_max = 104.5
label_min = 0.
judge_max = 10.
judge_min = 0.

# output dimension of I3D backbone
feature_dim = 1024

output_dim = {'MUSDL+LVFL': 21}


num_judges = 7
