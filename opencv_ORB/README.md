https://chatgpt.com/s/t_6ace374d755081918aa1c37f439d0ca0

opencv版结果
```
(base) zyh@zyh:~/code/zyh_slam$ /home/zyh/code/zyh_slam/opencv_ORB/build/opencv_orb
Feature extraction time: 191.668 ms
Feature matching time: 0.557182 ms
Keypoints in image A: 500
Keypoints in image B: 500
Matches before filtering: 500
Filtering threshold (2 * min_dist): 16
Matches after filtering: 23
max_dist: 93
min_dist: 8
Match visualization saved to: /home/zyh/code/zyh_slam/opencv_ORB/build/orb_matches.jpg
```

手动实现结果
```
(base) zyh@zyh:~/code/zyh_slam$ /home/zyh/code/zyh_slam/opencv_ORB/build/opencv_orb_manual
Feature extraction time: 3793.6 ms
Feature matching time: 119.957 ms
Keypoints in image A: 1500
Keypoints in image B: 1500
Matches passing Lowe ratio test: 284
RANSAC inliers: 179
max_dist: 74
min_dist: 19
Displayed matches: 179
Match visualization saved to: /home/zyh/code/zyh_slam/opencv_ORB/build/orb_manual_matches.jpg
Estimated homography:
[1.122342604436649, -0.006951540278914368, -341.9148680007517;
 0.04192877545601335, 1.068192628501569, -97.60797040409076;
 2.774625954292044e-05, 4.547180199716405e-06, 1]
```