# mmclassification-ros

This is a ROS package for 2D classification, which simply utilizes the toolbox [MMClassification](https://github.com/open-mmlab/mmclassification) of [OpenMMLab](https://openmmlab.com/).

## Requirements

- ROS Melodic
- Python 3.6+, PyTorch 1.3+, CUDA 9.2+ and [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

## To-Do List

- [x] Finish .launch file
  - [x] Check default ros image_topic for raw image (on our own system)
  - [x] decide whether to delete the image_view node: NO!

- [x] Finish srv file
  - [x] figure out the output format (string?): NO!
  - [x] use it!

- [ ] Finish script
  - [ ] finish line 46 checkpoint

- [x] generate CMakeLists.txt and package.xml

## ROS Interfaces

### params

- `~publish_rate`: the debug image publish rate. default: 1hz
- `~is_service`: whether or not to use service instead of subscribe-publish
- `~visualization`: whether or not to show the debug image

### topics

- `~debug_image`: publish the debug image
- `~objects`: publish the inference result, containing the classification
- `~image`: subscribe the input image. The default one is '/camera/rgb/image_raw' for astra_pro

## Example

If you are using the astra_pro camera, you could

```shell
source ~/mmcls/bin/activate # in the python3 virtual env
roslaunch mmclassification_ros mmclassification.launch
```

will continously do the inference on the camera image and publish the detection results
