import numpy as np
import imutils
import cv2

class check_test:
    def __init__(self):
        self.big_circles = np.array([0,0,0])
        self.small_circles = np.array([0,0,0])

    #全局阈值
    def threshold_demo(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        #直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        print("threshold value %s"%ret)
        #cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
        #cv2.imshow("binary0", binary)
        return binary

    #局部阈值
    def local_threshold(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        #自适应阈值化能够根据图像不同区域亮度分布，改变阈值
        binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
        cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
        #  cv2.imshow("binary1", binary)
        return binary
    #用户自己计算阈值
    def custom_threshold(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        h, w =gray.shape[:2]
        m = np.reshape(gray, [1,w*h])
        mean = m.sum()/(w*h)
        print("mean:",mean)
        ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
        #cv2.imshow("binary2", binary)
        return binary

    #判断是否为矩形，是则返回True，否则返回Flase
    def is_contour_bad(self,c):
	    # 近似轮廓
	    peri = cv2.arcLength(c, True)    #计算轮廓的周长
	    area=cv2.contourArea(c)      # 计算面积
	    #print(int(peri))
	    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

	    # 判断当前的轮廓是不是矩形,并且满足矩形周长范围在(80,100)  面积在(400,500)
	    return  ((not len(approx) == 4) and (peri > 85) and (peri < 110) and area >400 and area < 500)

    # 找矩形
    def find_retangles(self,image,all_cnts):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #灰度化
        #edged = cv2.Canny(gray, 50, 100)
        #cv2.imshow("gray_img",gray)
        mask = np.ones(image.shape[:2],dtype="uint8")*255    #生成一个与图片同像素的全白图片

        # 在仅有黑色形状的图片中，寻找图中的轮廓(用于过滤圆形)
        cnts2 = cv2.findContours(gray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts2 = imutils.grab_contours(cnts2)

        # 循环遍历所有的轮廓，在mask中绘制出选中的轮廓
        for c in all_cnts:
        # draw the contour and show it
        #peri2 = cv2.arcLength(c, True)    #计算轮廓的周长
            area=cv2.contourArea(c)
        #print(area)
            if area==0:
                continue
            if (area > 80) and (area < 120):
                cv2.drawContours(mask, [c], -1, (0,0,0), 2)   #描黑块

        cv2.imshow("checked_img",mask)
        # 循环遍历所有的轮廓,筛选出为矩形的
        for c in cnts2:
        # 检测该轮廓的类型，在新的mask中绘制结果
            if self.is_contour_bad(c):
                cv2.drawContours(mask,[c],-1,0,1)
	            # 移除不满足条件的轮廓并在白色图片mask中用黑色笔画出来
        cv2.imshow("find_rectangle",mask)
        #mask=cv2.Canny(mask,50,100)

        #检测矩形
        juxing_cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        juxing_cnts = imutils.grab_contours(juxing_cnts)
        print("I found {} black rectangle shapes".format(len(juxing_cnts)))

        #在原图上选中的矩形中绘制矩形框
        #rectangles ->存储选中矩形的的矩阵
        rectangles=np.array([[0,0]])
        for c in juxing_cnts:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)  #画矩形框
            cv2.circle(image,(int(x+w*0.5),int(y+h*0.5)),2,(255,255,255),1)     #画出中心坐标
            rectangles=np.concatenate((rectangles,[[int(x+w*0.5),int(y+h*0.5)]]),axis=0)

        rectangles=rectangles[np.lexsort(rectangles[:,:2:].T)]  #对矩形矩阵的y坐标排序
        #cv2.imshow('draw_rectangle',image)
        #print("选中的矩形矩阵坐标(按y坐标排序)\n","   x    y\n",rectangles[1:,:])

    # 找圆形
    def find_circles(self,image):
        imgray=cv2.Canny(image,800,1500)#Canny算子边缘检测
        #def canny(image,threshold1,threshold2)
        #第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
        #第二个参数是阈值1；
        #第三个参数是阈值2。


        self.big_circles = cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,minDist=23,param1=20,param2=10,minRadius=8,maxRadius=11)   #半径范围：8-11检测大圆,3-7检测小圆
        self.small_circles = cv2.HoughCircles(imgray,cv2.HOUGH_GRADIENT,1,minDist=23,param1=20,param2=8,minRadius=4,maxRadius=7)
#霍夫圆变换  (返回一个圆心坐标和半径的矩阵)
#def HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)   #这是HoughCircles在python中的定义
#@param image 8-bit, single-channel, grayscale input image.  image输入必须是8位的单通道灰度图像
#@param circles Output vector of found circles.Each vector is encoded as 3 or 4 element circle也就是我们最后圆的结果集
#@param method 目前只有HOUGH_GRADIENT,也就是2-1霍夫变换
#@param dp 原图像和累加器juzh矩阵的像素比 一般设为1就可以了
#@param minDist 圆心center中圆心之间的最小圆心距 如果小于此值,则认为两个是同一个圆(此时抛弃该圆心点,防止圆过于重合)
#@param param1 canny双阈值边缘检测的高阈值,经查阅一般低阈值为高阈值的0.4或者0.5
#@param param2 在确定圆心时 圆周点的梯度的累加器投票数ddata以及在确定半径时相同圆心相同半径的圆的个数max_count必须大于此阈值才认为是合法的圆心和半径
#@param minRadius 最小的半径 如果过小可能会因为噪音导致找到许多杂乱无章的圆,过大会找不到圆
#@param minRadius 最大的半径 如果过小可能会找不到圆,过大会找很多杂乱无章的圆


        self.big_circles = np.uint16(np.around(self.big_circles))
        #print(big_circles)
        self.small_circles = np.uint16(np.around(self.small_circles))
        #np.uint16数组转换为16位，表示范围0-65535
        #np.around返回四舍五入后的值

        Big_circles = self.big_circles[0]  #去掉circles数组一层外(中)括号
        Small_circles = self.small_circles[0]

        #过滤掉横坐标不在指定范围内的大圆(题目)
        Small_circles=Small_circles[Small_circles[:,0]<image.shape[0]/10]   #第一步过滤，x坐标小于图片宽度的10分之1

        Small_circles_avg_x=(np.sum(Small_circles[:,0]))/len(Small_circles)   #第二步过滤，小于或等于所有（题目）x坐标的平均值
        Small_circles=Small_circles[Small_circles[:,0]<=Small_circles_avg_x]


        #画大圆(选项)
        for i in Big_circles:
            # 画出外圆
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),3)#第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细
            # 画出圆心
            cv2.circle(image,(i[0],i[1]),2,(255,255,255),1)   #255,255,255是白色  0,255,0是绿色
        #画小圆(题目)
        for i in Small_circles:
            # 画出外圆
            cv2.circle(image,(i[0],i[1]),i[2],(0,255,0),3)#第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细
            # 画出圆心
        cv2.circle(image,(i[0],i[1]),2,(255,255,255),1)   #255,255,255是白色

        cv2.imshow('Draw_circles',image)
        print("大圆(选项)的个数是：",len(Big_circles))
        print("小圆(题目)的个数是：",len(Small_circles))

        #对ndarray类型的数据排序   小圆是选项，大圆是题目
        #print(Small_circles)
        Small_circles=Small_circles[np.lexsort(Small_circles[:,:2:].T)]  #对小圆数据矩阵按第二列排序(纵坐标排序)
        Big_circles=Big_circles[np.lexsort(Big_circles[:,:2:].T)]  #对大圆数据矩阵按第二列排序(纵坐标排序)
        return Small_circles,Big_circles

# def main():
#     check_tool = check_test()     # 初始化类
#     image = cv2.imread("D:\\Englishtest8.jpg")
#     image_two = check_tool.threshold_demo(image)   #二值化处理
#
#     opposite_image = 255-image_two   #对图像进行反相处理，即黑白颜色调换
#
#     # 在opposite_image中寻找轮廓
#     all_cnts = cv2.findContours(opposite_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     all_cnts = imutils.grab_contours(all_cnts)
#
#     #过滤掉圆,找到选中矩形框
#     check_tool.find_retangles(opposite_image,all_cnts)
#
#     #查找圆形
#     check_tool.find_circles()

if __name__ == '__main__':
    check_tool = check_test()     # 初始化类
    image = cv2.imread("D:\\Englishtest8.jpg")
    image_two = check_tool.threshold_demo(image)   #二值化处理

    opposite_image = 255-image_two   #对图像进行反相处理，即黑白颜色调换

    # 在opposite_image中寻找轮廓
    all_cnts = cv2.findContours(opposite_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    all_cnts = imutils.grab_contours(all_cnts)

    #过滤掉圆,找到选中矩形框
    check_tool.find_retangles(opposite_image,all_cnts)

    #查找圆形
    check_tool.find_circles()
