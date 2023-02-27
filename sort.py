"""
  SORT: A Simple, Online and Realtime Tracker
  关键步骤：
    --> 1，卡尔曼滤波预测出预测框
    --> 2，使用匈牙利算法将卡尔曼滤波的预测框和yolo的检测框进行IOU匹配来计算相似度 
    --> 3，卡尔曼滤波使用yolo的检测框更新卡尔曼滤波的预测框
  代码逐行注释，https://blog.csdn.net/zimiao552147572/article/details/106009225
"""
import os
import numpy as np
np.random.seed(0)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter


class KalmanPointTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as point.
  假设相邻两次观测的时间差都相同。
  """
  def __init__(self, point=[0,0]):
    # Initialises a tracker using initial point.
    self.kf = KalmanFilter(dim_x=4, dim_z=2)  # 4：状态量数目，包括（x，y，vx，vy）坐标及速度（每次移动的距离）；2：观测量数目，能看到的是坐标值
    dt = 1
    self.kf.F = np.array([[1,0,dt,0],
                          [0,1,0,dt],
                          [0,0,1,0],
                          [0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0],
                          [0,1,0,0]])
    # R是测量噪声的协方差矩阵，2x2，即真实值与测量值差的协方差
    # P是先验估计的协方差矩阵，4x4
    # Q是过程噪声的协方差矩阵，4x4
    self.kf.P[2:, 2:] *= 1000.  # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.R = 1.  # 测量噪声矩阵的权重值
    # self.kf.Q[-1,-1] *= 0.01  # 过程噪声矩阵
    # self.kf.Q[2:,2:] *= 0.01
    self.kf.Q *= 0.003
    # 状态更新向量x(状态变量x)设定是一个四维向量：x=[x，y，vx，vy].T。
    self.kf.x[:2] = np.array(point).reshape(2, 1)  # 用第一个点的坐标初始化x

    self.history = []  # 保存单个目标框连续预测的多个结果到history列表中

  def update(self, point):
    self.kf.update(point)

  def predict(self):
    self.kf.predict()
    return self.kf.x


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  假设相邻两次观测的时间差都相同。
  """
  count = 0
  def __init__(self, bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    # define constant velocity model
    # dim_x=7定义是一个7维的状态更新向量x(状态变量x)：x=[u,v,s,r,u^,v^,s^]T。
    # dim_z=4定义是一个4维的观测输入，即中心面积的形式[x,y,s,r]，即[检测框中心位置的x坐标,y坐标,面积,宽高比]。
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    # 通过4*7的量测矩阵H(观测矩阵H) 乘以 7*1的状态更新向量x(状态变量x) 即可得到一个 4*1的[u,v,s,r]的估计值。
    self.kf.F = np.array([[1,0,0,0,1,0,0],
                          [0,1,0,0,0,1,0],
                          [0,0,1,0,0,0,1],
                          [0,0,0,1,0,0,0],
                          [0,0,0,0,1,0,0],
                          [0,0,0,0,0,1,0],
                          [0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],
                          [0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,0],
                          [0,0,0,1,0,0,0]])

    # R是测量噪声的协方差矩阵，4x4，即真实值与测量值差的协方差
    # P是先验估计的协方差矩阵，7x7
    # Q是过程噪声的协方差矩阵，7x7
    self.kf.R[2:,2:] *= 10.  # 测量噪声矩阵
    self.kf.P[4:,4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01  # 过程噪声矩阵
    self.kf.Q[4:,4:] *= 0.01

    # 状态更新向量x(状态变量x)设定是一个七维向量：x=[u,v,s,r,u^,v^,s^]T。
    self.kf.x[:4] = self.convert_bbox_to_z(bbox)  # 表示 u、v、s、r初始化为第一帧bbox观测到的结果[x,y,s,r]
    self.time_since_update = 0  # 当前连续预测的次数，只要调用KalmanBoxTracker类中update函数，time_since_update就会清零
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    # 保存单个目标框连续预测的多个结果到history列表中，一旦执行update就会清空history列表。
    # 会将预测的候选框从中心面积的形式[x,y,s,r]转换为坐标的形式[x1,y1,x2,y2] 的bbox 再保存到 history 列表中。
    self.history = []  
    self.hits = 0  # 该目标框进行更新的总次数。每执行update一次，便hits+=1。
    self.hit_streak = 0  # 连续更新的次数。判断当前是否做了更新，大于等于1的说明做了更新，只要连续帧中没有做连续更新，hit_streak就会清零
    self.age = 0  # 该目标框进行预测的总次数。每执行predict一次，便age+=1。

  def update(self, bbox):
    """
    用检测框替换跟踪器self.trackers列表中对应的跟踪框
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0  # 重置为0
    self.history = []  # 重置为空
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(self.convert_bbox_to_z(bbox))

  def predict(self):
    """
    用卡尔曼滤波对跟踪器列表中的目标进行下一时刻位置的预测
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    # 若过程中未更新过，将hit_streak置为0
    miss_det_num = 2  # 允许跟丢的最大次数，默认值是1帧
    if(self.time_since_update >= miss_det_num):
      self.hit_streak = 0
    self.time_since_update += 1
    # 把目标框当前该次的预测的结果([x,y,s,r]转换后的[x1,y1,x2,y2])进行返回输出。
    self.history.append(self.convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    坐标转换，Returns the current bounding box estimate.
    """
    return self.convert_x_to_bbox(self.kf.x)

  def convert_bbox_to_z(self, bbox):
    """
    将[x1,y1,x2,y2]形式的检测框转为滤波器的状态表示形式[x,y,s,r]。
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

  def convert_x_to_bbox(self, x, score=None):
    """
    从归一化坐标转换为像素坐标
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
      return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
      return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    输入包含：
      1.连续检测 N次才会被视为命中
      2.丢失M帧会被视为track丢失
      3.前后帧匹配的条件，比如IOU，中心点距离
    """
    self.max_age = max_age  # 连续预测的最大次数，即目标未被检测到的帧数，超过之后会被删
    # min_hits:最小更新的次数，就是放在self.trackers跟踪器列表中的框与检测框匹配上，然后调用卡尔曼滤波器类中的update函数的最小次数，
    # min_hits不设置为0是因为第一次检测到的目标不用跟踪，只需要加入到跟踪器列表中，不会显示，这个值不能设大，一般就是1，表示如果连续两帧都检测到目标
    self.min_hits = min_hits  # 目标命中的最小次数，小于该次数update函数不返回该目标的KalmanBoxTracker卡尔曼滤波对象
    self.iou_threshold = iou_threshold
    self.trackers = []  # 维护所有的跟踪序列，列表元素是KalmanBoxTracker的对象
    self.frame_count = 0  # 帧计数

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # 初始化，get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []  # 存储要删除的目标框
    ret = []  # 存储要返回的追踪目标框
    # 遍历跟踪序列，在trk中记录卡尔曼滤波器predict的bbox结果
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]  # 调用卡尔曼滤波器预测在当前帧中的位置
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      # 如果跟踪框中包含空值则将该跟踪框添加到要删除的列表中
      if np.any(np.isnan(pos)):
        to_del.append(t)
    # numpy.ma.masked_invalid 屏蔽出现无效值的数组（NaN 或 inf）
    # numpy.ma.compress_rows 压缩包含掩码值的2-D 数组的整行，将包含掩码值的整行去除
    # trks中存储了上一帧中跟踪的目标并且在当前帧中的预测跟踪框
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    # 逆向删除异常的跟踪器，防止破坏索引
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    # 将跟踪序列和当前帧的检测结果进行数据关联
    matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])  # 将跟踪成功的物体BBox信息更新到对应的卡尔曼滤波器状态向量

    # create and initialise new trackers for unmatched detections
    # 新增的物体要创建新的卡尔曼滤波器用于跟踪
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    
    # 自后向前遍历，仅返回在当前帧出现且命中周期大于self.min_hits（除非跟踪刚开始）的跟踪结果；如果未命中时间大于self.max_age则删除跟踪器。
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        # self.min_hits不设置为0是因为第一次检测到的目标不用跟踪，不能设大，一般就是1
        time_window = 1  # 表示连续预测的次数
        # 跟踪成功目标的box与id放入ret列表中
        if (trk.time_since_update < time_window) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    
    # 返回当前画面中所有被跟踪物体的BBox与ID，二维矩阵[[x1,y1,x2,y2,track_id],,,[,,,]]
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))

  def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.3):
    """
    # 线性分配（匈牙利算法），将物体检测的BBox与卡尔曼滤波器预测的跟踪BBox匹配
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    matches：是m*2的矩阵，每一行元素是：[目标检测框的索引，跟踪序列的索引]
    """
    # 第一帧没有跟踪框，只有检测框，所以返回3个值：（1）匹配到的[d,t]（空的）；（2）没有匹配到的检测框；（3）没有匹配到的跟踪框（空的）
    if(len(trackers)==0):
      return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    # 根据IOU计算代价矩阵，行是det，列是track
    iou_matrix = self.iou_batch(detections, trackers)

    # 通过匈牙利算法匹配卡尔曼滤波器预测的BBox与物体检测BBox以[[d,t],,,]的二维矩阵保存到 matched_indices
    if min(iou_matrix.shape) > 0:
      a = (iou_matrix > iou_threshold).astype(np.int32)
      if a.sum(1).max() == 1 and a.sum(0).max() == 1:
          matched_indices = np.stack(np.where(a), axis=1)
      else:
        matched_indices = self.linear_assignment(-iou_matrix)
    else:
      matched_indices = np.empty(shape=(0,2))

    # 没有匹配上的物体检测BBox放入 unmatched_detections 列表，表示有新的物体进入画面了，后面要新增跟踪器追踪新物体
    unmatched_detections = []
    for d, det in enumerate(detections):
      if(d not in matched_indices[:,0]):
        unmatched_detections.append(d)
    # 没有匹配上的卡尔曼滤波器预测的BBox放入 unmatched_trackers 列表，表示之前跟踪的物体离开画面了，后面可能要删除对应的跟踪器
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
      if(t not in matched_indices[:,1]):
        unmatched_trackers.append(t)

    # 遍历 matched_indices 矩阵，将IOU值小于 iou_threshold 的匹配结果分别放入 unmatched_detections，unmatched_trackers 列表中
    matches = []
    for m in matched_indices:
      if(iou_matrix[m[0], m[1]] < iou_threshold):
        unmatched_detections.append(m[0])
        unmatched_trackers.append(m[1])
      else:
        matches.append(m.reshape(1,2))
    # 匹配上的卡尔曼滤波器预测的BBox与物体检测的BBox以[[d,t],,,]的形式放入matches矩阵
    if(len(matches)==0):
      matches = np.empty((0,2),dtype=int)
    else:
      matches = np.concatenate(matches,axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

  def linear_assignment(self, cost_matrix):
    try:
      import lap
      _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
      return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
      from scipy.optimize import linear_sum_assignment
      x, y = linear_sum_assignment(cost_matrix)
      return np.array(list(zip(x, y)))

  def iou_batch(self, bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
    return (o)  


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=2)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  mot_dir = '/home/pd_mzc/Documents/dangerous-scene-recognition/src/sort/data/'
  if(display):
    if not os.path.exists(mot_dir):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    # plt.ion()
    fig = plt.figure(figsize=(6,5), dpi=200, constrained_layout=True)
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  # 遍历每个数据集
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)
    # 读取目标检测结果。det.txt中每一行表示一个对象，第一列表示所在的帧序号，一帧中可能有多个对象
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    start_no = 100000
    # 会将跟踪结果写入txt文件
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      # 逐帧遍历
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #detection and frame numbers begin at 1
        # 根据帧序号提取当前帧中所有检测到的对象
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]  # x1,y1,w,h,conf
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        # (可选)读取每帧对应的图片进行可视化
        if(display):
          fn = os.path.join(mot_dir, phase, seq, 'img1', '%06d.jpg'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets')

        # 使用当前帧的检测结果更新全局跟踪器
        start_time = time.time()
        trackers = mot_tracker.update(dets)  # trackers：np.array, n*5， [x1,y1,x2,y2,track_id]
        cycle_time = time.time() - start_time
        total_time += cycle_time

        # 保存跟踪的结果
        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=2,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          # plt.draw()
          plt.savefig("./output/{}.jpg".format(start_no+frame))
          print("save:", start_no+frame)
          start_no += 1
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
