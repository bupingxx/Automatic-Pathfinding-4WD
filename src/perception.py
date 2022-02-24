"""perception controller."""

from shutil import move
from controller import Robot
from controller import Camera
from controller import Motor
from controller import GPS
from controller import Keyboard
from controller import InertialUnit
from controller import Lidar
from scipy import interpolate

import numpy as np
import math
import random
import copy
import cv2 as cv
DOUBLE_MAX = 1.7976931348623158E+308

# 手动输入
step = 0.8                      # 步长     1号地图1.0  2号地图0.8 3号地图0.8
way = 2.7                    # 临界距离 1号地图3.5  2号地图1.4   3号地图2.5
#begin = [2.0, 2.0]            # 1号地图起点坐标
#end = [-2.0, -2.0]            # 1号地图终点坐标
#begin = [2.0, 2.5]           # 2号地图起点坐标
#end = [-2.0, -2.5]           # 2号地图终点坐标
begin = [4.5, 4.5]           # 3号地图起点坐标
end = [-4.5, -4.5]           # 3号地图终点坐标

#小车速度
velocity = 3
speed_forward = [2*velocity, 2*velocity, 2*velocity, 2*velocity]
speed_backward = [-2*velocity, -2*velocity, -2*velocity, -2*velocity]
speed_leftCircle = [velocity, -velocity, -velocity, velocity]
speed_rightCircle = [-velocity, velocity, velocity, -velocity]
speed1 = [0, 0, 0, 0]
speed2 = [0, 0, 0, 0]

# 地图信息
 
map_proportion = 0.01       # 地图映射的比例
numPoints = 3              # 采样点数量
throwfail = 50             # 采样上限
accuracy = 100              # 碰撞检测精度
points = []                 # 顶点集合
path = []                   # 最短路径 
visited = [0, 1]            # 已遍历顶点
outlierCnt = 3             # 离群点检测范围
grow = 1                   # 生长方向
first = 1

# 通过映射之后的地图大小
map_size_x = 10             # 设置地图大小
map_size_y = 10             # 设置地图大小
grid_x = (int)(map_size_x / map_proportion)
grid_y = (int)(map_size_y / map_proportion)
grid_map = np.zeros([grid_y,grid_x],np.float32)
path_map = np.zeros([grid_y,grid_x],np.float32)
# 转换起点和终点
start = [map_size_x/2 - begin[0], map_size_y/2 - begin[1], -1, -1] # x,y坐标, 父节点, 以及到终点代价
finish = [map_size_x/2 - end[0], map_size_y/2 - end[1], -1, -1]

#规划时使用的参数
count = 1
halt = 1            # 清空地图，等待重新建图的时间
rrt_fail_flag = 0   # rrt失败
still_cnt = 0       # 长时间停留在原地
move_flag = 0       # 1时控制小车移动
q_flag = 0          # 左转
e_flag = 0          # 右转
last_dis = DOUBLE_MAX
now_dis = 0
dir = 1
move_count = 0
idx = 0             # 搜索的顶点下标
move_idx = 0        # 小车沿着路径走的下标

# 将空间中的坐标转换为地图坐标
def turn(x):
    return (int)((x / map_proportion))

# 两点间距离
def Cost(x1, y1, x2, y2):
    enddis = math.sqrt(pow(abs(x1-x2), 2) + pow(abs(y1-y2), 2))
    return enddis

def double_check(cur_x, cur_y):
    for i in range(3, len(points)):
        if(Cost(cur_x, cur_y, points[i][0], points[i][1]) < step/3):
            return False
    return True

# 碰撞检测
def reachLine(x1, y1, x2, y2, maze, need_double_check):
    dx = (float)(x1 - x2)/(float)(accuracy)
    dy = (float)(y1 - y2)/(float)(accuracy)

    t = 0.04
    for i in range(accuracy+1):
        x = x1 - dx*i
        y = y1 - dy*i
        #print("check", turn(x), turn(y))
        if( x-t > 0 and x+t < map_size_x and y-t > 0 and y+t < map_size_y
        and maze[turn(y)][grid_x - 1 - turn(x-t)]!=1 and maze[turn(y)][grid_x - 1 - turn(x+t)]!=1
        and maze[turn(y-t)][grid_x - 1 - turn(x)]!=1 and maze[turn(y+t)][grid_x - 1 - turn(x)]!=1 
        and maze[turn(y)][grid_x - 1 - turn(x)]!=1):
            continue
        else:
            return False
    if need_double_check == 1:
        if double_check(x1, y1):
            return True
        else:
            return False
    return True

# 将采样点加入树，如果可达终点则返回true
def formTree(cur, cur_x, cur_y, x, y, maze):
    cost = Cost(x, y, finish[0], finish[1])
    points.append([x, y, cur, cost])
    flag = False


    if(reachLine(x, y, finish[0], finish[1], maze, 0) and Cost(x, y, finish[0], finish[1]) < step):
        points[1][2] = len(points)-1
        #cv.line(path_map, [grid_x-turn(x)-1, turn(y)], [grid_x-turn(finish[0])-1, turn(finish[1])], 1, 2)
        flag = True

    # 绘制搜索路径图
    # cv.namedWindow('maze', 0)
    # cv.resizeWindow('maze', 500, 500)
    # cv.line(path_map, [grid_x-turn(x)-1, turn(y)], [grid_x-turn(cur_x)-1, turn(cur_y)], 1, 2)
    # cv.imshow('maze', path_map)
    # cv.waitKey(100)
    return flag   


# 在当前节点周围随机生成节点， 舍弃在障碍物上的点
# 终点可达时返回True
def throwPoints(cur, num, maze):
    min = DOUBLE_MAX
    min_idx = 0
    cnt = 0
    cur_x = points[cur][0]
    cur_y = points[cur][1]
    # 根据随机数，选择向终点伸展或者是随机伸展
    random.seed()

    for i in range(num):
        while(1):
            if(cnt >= throwfail):       # 超过采样上限
                return min_idx
                
            agl = random.uniform(0.0, 2 * math.pi)
            # 根据随机数，选择向终点伸展或者是随机伸展
            # decide = random.random()
            # if( decide > 0.7):
            #    agl = random.uniform(-0.2 * math.pi, 0.7 * math.pi)
            if( cnt == 0):
                agl = grow * 0.5 * math.pi
            # elif (decide < 0.2):
            #     agl = random.uniform(math.pi, 1.5 * math.pi)

            x = cur_x + step * math.sin(agl)
            y = cur_y + step * math.cos(agl)
            cnt += 1
            if(reachLine(x, y, cur_x, cur_y, maze, 1)):
                i += 1
                if(Cost(x,y,finish[0],finish[1]) < min):
                    min = Cost(x,y,finish[0],finish[1])
                    min_idx = len(points)

                if(formTree(cur, cur_x, cur_y, x, y, maze)):
                    return -1
                else:
                    break
    # 返回最小代价的新节点
    return min_idx

# 查找列表
def isInList(target, list):
    for i in list:
        if(target == i):
            return True
    return False
    
# 合并路径
def join(maze):
    v = path[0]
    v_x = points[v][0]
    v_y = points[v][1]
    for i in range(len(path)-1, 1, -1):
        u = path[i]
        u_x = points[u][0]
        u_y = points[u][1]
        if(Cost(v_x, v_y, u_x, u_y) < way*step and reachLine(v_x, v_y, u_x, u_y, maze, 0)):
            for t in range(i-1):
                path.remove(path[1])
            print("after join:", path)
            return 

# 形成路径
def formPath():
    idx = 1
    while(points[idx][2]!=-1):
        path.append(idx)
        idx = points[idx][2]
    path.append(0)
    path.reverse()
    #print(path)

# RRT
def rrt(idx, maze): 
    cnt = 0
    fail_flag = 0
    while(cnt < 5000):
        visited.append(idx)
        res = throwPoints(idx, numPoints, maze)
        if(res == -1):
            #print("Success to find a path.")    # 成功找到路径
            formPath()
            idx = res
            break
        elif(res == 0):                         # 查找失败，回退到父节点重新生长
            idx = points[idx][2]
            if(idx == 0):
                fail_flag += 1
                if fail_flag == numPoints:      # 多次回退到根节点，说明此次RRT失败
                    return -2
        else:                                   # 搜索树下一层
            idx = res
        cnt += 1

    join(maze)

    if len(path) >= 2:
        1
        # # 打印全局地图
        # cv.namedWindow('map', 0)
        # cv.resizeWindow('map', 500, 500)
        # cv.line(maze, [grid_x-turn(finish[0])-1, turn(finish[1])], [grid_x-turn(points[path[1]][0])-1, turn(points[path[1]][1])], 1, 2)  
        # cv.imshow('map', maze)
        # cv.waitKey(100)
    return idx

# 检测进行测试的点是否在地图中
def check_in_range(a , start , end):
    if a < start:
       a = start
    elif a > end:
        a = end
    return a

# main
points.append(start)        # 起点
points.append(finish)       # 终点

robot = Robot()
timestep = int(robot.getBasicTimeStep())

#小车 电机 gps imu lidar 键盘
motor = []
for i in range(4):
    cur_motor:Motor
    cur_motor = robot.getDevice("motor"+str(i + 1))
    cur_motor.setPosition(float('inf'))
    cur_motor.setVelocity(0.0)
    motor.append(cur_motor)

gps:GPS
gps = robot.getDevice("car_gps")
gps.enable(timestep)

imu:InertialUnit
imu = robot.getDevice("imu")
imu.enable(timestep)

lidar:Lidar
lidar = robot.getDevice("lidar")
lidar.enable(timestep)

camera:Camera
camera = robot.getDevice("camera")
camera.enable(timestep)

keyboard:Keyboard
keyboard = Keyboard()
keyboard.enable(timestep)

#设定两帧的雷达之间的坐标
last_pos = [0.0, 0.0, 0.0]
now_pos = [0.0, 0.0, 0.0]

while robot.step(timestep) != -1:
    count += 1
    #计算小车当前在构建的图中的坐标
    rpy = imu.getRollPitchYaw()
    roll_angle = rpy[0] - math.pi/2
    gps_pos = gps.getValues()
    gps_x = gps_pos[0]
    gps_y = gps_pos[1]
    gps_x = map_size_x/2 - gps_x  
    gps_y = map_size_y/2 - gps_y

    #整合朝向与坐标
    last_pos = now_pos
    now_pos = [gps_x, gps_y, roll_angle]
    if count == 0:
        last_pos = now_pos
    
    #雷达的使用  lidar_range
    lidar_range = lidar.getRangeImage()
    
    #处理雷达的运动畸变
    X_range = [0,359]
    X_pos = np.arange(0,360,1)
    inter_x = [last_pos[0], now_pos[0]]
    inter_y = [last_pos[1], now_pos[1]]
    inter_roll = [last_pos[2], now_pos[2]]
    
    f_x = interpolate.interp1d(X_range , inter_x)
    Res_X = f_x(X_pos)
    f_y = interpolate.interp1d(X_range , inter_y)
    Res_Y = f_y(X_pos)
    f_Roll = interpolate.interp1d(X_range , inter_roll)
    Res_Roll = f_Roll(X_pos)
    
    #将雷达遇到的障碍物写入地图，黑色的为1
    if count % 5 == 0 or (halt > 0 and halt % 2 == 0):
        for i in range(360):
            if lidar_range[i] == 0 or lidar_range[i] == float('inf') :
                continue
            
            # 离群点检测
            outlierCheck = 0.0
            for k in range(i - outlierCnt, i + outlierCnt):
                if (k >=0 and k <360):
                    outlierCheck += abs(lidar_range[i] - lidar_range[k])
            outlierCheck /= (2 * outlierCnt)
            if (outlierCheck > 0.1 * lidar_range[i]):
                continue

            angle = Res_Roll[i]  - math.pi*i/180
            x = Res_X[i] + lidar_range[i] * math.cos(angle)
            y = Res_Y[i] + lidar_range[i] * math.sin(angle)
            
            x = np.fix(x / map_proportion)
            y = np.fix(y / map_proportion)
            x = (int)(check_in_range(x , 0 , grid_x-1))
            y = (int)(check_in_range(y , 0 , grid_y-1))
            grid_map[ y , grid_x - x -1 ] = 1.0

    # rrt路径搜索
    if(count % 20 == 0 and move_flag == 0 and idx != -1 and halt == 0):
        tmp_map = copy.copy(grid_map)
        #kernel = np.ones((20, 20), dtype=np.uint8)
        #tmp_map = cv.dilate(tmp_map, kernel, 1)  # 1:迭代次数，也就是执行几次膨胀操作
        idx = rrt(idx, tmp_map)
        if idx == -1:
            move_flag = 1
            rrt_fail_flag = 0
        else:       #rrt失败时清理地图
            print("rrt fail, clear map")
            rrt_fail_flag = 1
            halt = 1
            grid_map = np.zeros([grid_y,grid_x],np.float32)
            path = []
            move_idx = 0
            idx = 0
            visited = [0, 1]
            start = [gps_x, gps_y, -1, -1]
            points = []
            points.append(start)        # 起点
            points.append(finish)       # 终点
            move_flag = 0
            path_map = np.zeros([grid_y,grid_x],np.float32)

    for i in range(4):
        speed1[i] = 0
    # rrt后控制小车移动
    if(move_flag):
        # 卡墙检测
        # if(Cost(now_pos[0], now_pos[1], last_pos[0], last_pos[1])<0.01):
        #     still_cnt += 1
        #     if(still_cnt > 500):
        #         still_cnt = 0
        #         move_flag = 0
        #         clear(now_pos[0], now_pos[1])
        #         continue
        # else:
        #     still_cnt = 0
        
        move_count += 1
        err = 0.15
        cur_x = gps_x
        cur_y = gps_y
        to_x = points[path[move_idx]][0]
        to_y = points[path[move_idx]][1]
        agl = (math.atan( (cur_y - to_y) / (cur_x - to_x)) + 2*math.pi) % (2*math.pi)
        car_agl = (rpy[0]- math.pi/2 + 2*math.pi) % (2*math.pi)
        now_dis = Cost(cur_x, cur_y, to_x, to_y)
        if(abs(car_agl - agl)< 0.02):
            q_flag = 0
            e_flag = 0
            if(now_dis <= last_dis):
                dir = dir
            else:
                dir = -dir
            for i in range(4):
                speed1[i] =  dir*speed_forward[i]
        elif(q_flag == 1 or car_agl < agl):
            q_flag = 1
            for i in range(4):
                speed1[i] = speed_leftCircle[i]
        elif(e_flag == 1 or car_agl > agl):
            e_flag = 1
            for i in range(4):
                speed1[i] = speed_rightCircle[i]

        # 到达终点
        if (Cost(gps_x, gps_y, finish[0], finish[1]) < 0.15):
            for i in range(4):
                motor[i].setVelocity(0)
            print("Arrive!")
            break

        # 到达目的点
        if(abs(cur_x - to_x) < err and abs(cur_y - to_y) < err):
            move_idx += 1

        # 到达当前已知路径的终点
        if((move_idx == 2 or move_idx == len(path)-1) and len(path)!=2):
            #print("arrive, clear anr rrt")
            path = []
            move_idx = 0
            idx = 0
            visited = [0, 1]
            start = [cur_x, cur_y, -1, -1]
            points = []
            points.append(start)        # 起点
            points.append(finish)       # 终点
            move_flag = 0
            path_map = np.zeros([grid_y,grid_x],np.float32)
        elif rrt_fail_flag == 0 and move_idx == len(path):
            print("Arrive!")
            break

        last_dis = now_dis

    # 旋转探测周围
    if (halt > 0 and halt < 100):
        halt += 1
        for i in range(4):
            speed1[i] = 0.5 * speed_rightCircle[i]
    else:
        halt = 0

    for i in range(4):
        motor[i].setVelocity(speed1[i])


for i in range(4):
    motor[i].setVelocity(0)
# cv.namedWindow('maze', 0)
# cv.resizeWindow('maze', 500, 500)
# cv.imshow('maze', path_map)
# cv.waitKey(0)
# cv.destroyAllWindows()
# Enter here exit cleanup code.
