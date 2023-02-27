SORT
=====

refer:https://github.com/abewley/sort

A simple online and realtime tracking algorithm for 2D multiple object tracking in video sequences.
See an example [video here](https://alex.bewley.ai/misc/SORT-MOT17-06-FRCNN.webm).

By Alex Bewley  

### Introduction

SORT is a barebones implementation of a visual multiple object tracking framework based on rudimentary data association and state estimation techniques. It is designed for online tracking applications where only past and current frames are available and the method produces object identities on the fly. While this minimalistic tracker doesn't handle occlusion or re-entering objects its purpose is to serve as a baseline and testbed for the development of future trackers.

SORT was initially described in [this paper](http://arxiv.org/abs/1602.00763). At the time of the initial publication, SORT was ranked the best *open source* multiple object tracker on the [MOT benchmark](https://motchallenge.net/results/2D_MOT_2015/).

**Note:** A significant proportion of SORT's accuracy is attributed to the detections.
For your convenience, this repo also contains *Faster* RCNN detections for the MOT benchmark sequences in the [benchmark format](https://motchallenge.net/instructions/). To run the detector yourself please see the original [*Faster* RCNN project](https://github.com/ShaoqingRen/faster_rcnn) or the python reimplementation of [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) by Ross Girshick.

**Also see:**
A new and improved version of SORT with a Deep Association Metric implemented in tensorflow is available at [https://github.com/nwojke/deep_sort](https://github.com/nwojke/deep_sort) .




### Dependencies:

To install required dependencies run:
```
$ pip install -r requirements.txt
```


### Demo:

To run the tracker with the provided detections:

```
$ cd path/to/sort
$ python sort.py
```

To display the results you need to:

1. Download the [2D MOT 2015 benchmark dataset](https://motchallenge.net/data/MOT15/)
0. Create a symbolic link to the dataset
  ```
  $ ln -s /path/to/MOT2015_challenge/data/2DMOT2015 mot_benchmark
  ```
0. Run the demo with the ```--display``` flag
  ```
  $ python sort.py --display
  ```

### Using SORT in your own project

Below is the gist of how to instantiate and update SORT. See the ['__main__'](https://github.com/abewley/sort/blob/master/sort.py#L239) section of [sort.py](https://github.com/abewley/sort/blob/master/sort.py#L239) for a complete example.
    
    from sort import *
    
    #create instance of SORT
    mot_tracker = Sort() 
    
    # get detections
    ...
    
    # update SORT
    track_bbs_ids = mot_tracker.update(detections)
    
    # track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
    ...

 

