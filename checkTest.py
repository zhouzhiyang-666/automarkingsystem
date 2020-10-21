import numpy as np
import imutils
import cv2
#  定义一个类，包含所有属性和方法
class check_test(object):
    def __init__(self,image):
        self.big_circles = np.array([0,0,0])
        self.small_circles = np.array([0,0,0])
        self.Big_circles = np.array([0, 0, 0])
        self.Small_circles = np.array([0, 0, 0])
        self.image = cv2.imread(image)

    # 全局阈值
    def threshold_demo(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割。
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        print("threshold value %s"%ret)
        # cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
        # cv2.imshow("binary0", binary)
        return binary

    # 局部阈值
    def local_threshold(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        # 自适应阈值化能够根据图像不同区域亮度分布，改变阈值
        binary =  cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, 10)
        cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
        #  cv2.imshow("binary1", binary)
        return binary
    # 用户自己计算阈值
    def custom_threshold(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #把输入图像灰度化
        h, w =gray.shape[:2]
        m = np.reshape(gray, [1,w*h])
        mean = m.sum()/(w*h)
        print("mean:",mean)
        ret, binary =  cv2.threshold(gray, mean, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("binary2", cv2.WINDOW_NORMAL)
        # cv2.imshow("binary2", binary)
        return binary

    # 判断是否为矩形，是则返回True，否则返回 Flase
    def is_contour_bad(self, c):
	    # 近似轮廓
	    peri = cv2.arcLength(c, True)    # 计算轮廓的周长
	    area=cv2.contourArea(c)      # 计算面积
	    #print(int(peri))
	    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 判断当前的轮廓是不是矩形,并且满足矩形周长范围在(80,100)  面积在(400,500)
	    return ((not len(approx) == 4) and (peri > 85) and ( peri < 110 ) and area > 400 and area < 500)

    # 找矩形
    def find_retangles(self):
        image_two = check_tool.threshold_demo(self.image)  # 二值化处理
        opposite_image = 255 - image_two  # 对图像进行反相处理，即黑白颜色调换
        cv2.imshow("oppositeImg",opposite_image)
        # 在二值化图片中寻找轮廓
        all_cnts = cv2.findContours(opposite_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_cnts = imutils.grab_contours(all_cnts)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)      #灰度化
        # edged = cv2.Canny(gray, 50, 100)
        # cv2.imshow("gray_img",gray)
        mask = np.ones(self.image.shape[:2],dtype="uint8")*255    # 生成一个与图片同像素的全白图片

        # # 在仅有黑色形状的图片中，寻找图中的轮廓(用于过滤圆形)
        # cnts2 = cv2.findContours(opposite_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # cnts2 = imutils.grab_contours(cnts2)

        # 循环遍历所有的轮廓，在mask中绘制出选中的轮廓
        for c in all_cnts:
        # draw the contour and show it
            peri = cv2.arcLength(c, True)    # 计算轮廓的周长
            area=cv2.contourArea(c)
        # print(area)
            if area==0:
                continue
            if (peri > 80) and (peri < 120) and (area > 380) and (area <520):
                cv2.drawContours(mask, [c], -1, (0,0,0), thickness=-1)   #描黑块
        cv2.imshow("checked_img" ,mask)
        canny_mask = cv2.Canny(mask,50,100)
        # 检测矩形
        juxing_cnts = cv2.findContours(canny_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        juxing_cnts = imutils.grab_contours(juxing_cnts)
        print("I found {} black rectangle shapes".format(len(juxing_cnts)))

        # 在原图上选中的矩形中绘制矩形框
        # rectangles ->存储选中矩形的的矩阵
        rectangles=np.array([[0,0]])
        for c in juxing_cnts:
            x,y,w,h=cv2.boundingRect(c)
            cv2.rectangle(self.image,(x,y),(x+w,y+h),(0,255,0),2)  #画矩形框
            cv2.circle(self.image,(int(x+w*0.5),int(y+h*0.5)),2,(255,255,255),1)     #画出中心坐标
            rectangles=np.concatenate((rectangles,[[int(x+w*0.5),int(y+h*0.5)]]),axis=0)

        rectangles = rectangles[np.lexsort(rectangles[:,:2:].T)]  #对矩形矩阵的y坐标排序
        cv2.imshow('draw_rectangle',self.image)
        # print("选中的矩形矩阵坐标(按y坐标排序)\n","   x    y\n",rectangles[1:,:])

    # 找圆形
    def find_circles(self):
        image_two = check_tool.threshold_demo(self.image)  # 二值化处理
        opposite_image = 255 - image_two  # 对图像进行反相处理，即黑白颜色调换
        cv2.imshow("oppositeImg", opposite_image)
        # def canny(image,threshold1,threshold2)
        # 第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
        # 第二个参数是阈值1；
        # 第三个参数是阈值2。


        self.big_circles = cv2.HoughCircles(opposite_image,cv2.HOUGH_GRADIENT,1,minDist=23,param1=20,param2=10,minRadius=8,maxRadius=11)   #半径范围：8-11检测大圆,3-7检测小圆
        self.small_circles = cv2.HoughCircles(opposite_image,cv2.HOUGH_GRADIENT,1,minDist=23,param1=20,param2=8,minRadius=4,maxRadius=7)
        # 霍夫圆变换  (返回一个圆心坐标和半径的矩阵)
        # def HoughCircles(image, method, dp, minDist, circles=None, param1=None, param2=None, minRadius=None, maxRadius=None)   #这是HoughCircles在python中的定义
        # @param image 8-bit, single-channel, grayscale input image.  image输入必须是8位的单通道灰度图像
        # @param circles Output vector of found circles.Each vector is encoded as 3 or 4 element circle也就是我们最后圆的结果集
        # @param method 目前只有HOUGH_GRADIENT,也就是2-1霍夫变换
        # @param dp 原图像和累加器juzh矩阵的像素比 一般设为1就可以了
        # @param minDist 圆心center中圆心之间的最小圆心距 如果小于此值,则认为两个是同一个圆(此时抛弃该圆心点,防止圆过于重合)
        # @param param1 canny双阈值边缘检测的高阈值,经查阅一般低阈值为高阈值的0.4或者0.5
        # @param param2 在确定圆心时 圆周点的梯度的累加器投票数ddata以及在确定半径时相同圆心相同半径的圆的个数max_count必须大于此阈值才认为是合法的圆心和半径
        # @param minRadius 最小的半径 如果过小可能会因为噪音导致找到许多杂乱无章的圆,过大会找不到圆
        # @param minRadius 最大的半径 如果过小可能会找不到圆,过大会找很多杂乱无章的圆


        self.big_circles = np.uint16(np.around(self.big_circles))
        # print(big_circles)
        self.small_circles = np.uint16(np.around(self.small_circles))
        # np.uint16数组转换为16位，表示范围0-65535
        # np.around返回四舍五入后的值

        self.Big_circles = self.big_circles[0]       # 去掉circles数组一层外(中)括号
        self.Small_circles = self.small_circles[0]

        # 过滤掉横坐标不在指定范围内的大圆(题目)
        self.Small_circles=self.Small_circles[self.Small_circles[:,0]<opposite_image.shape[0]/10]   #第一步过滤，x坐标小于图片宽度的10分之1

        self.Small_circles_avg_x=(np.sum(self.Small_circles[:,0]))/len(self.Small_circles)   #第二步过滤，小于或等于所有（题目）x坐标的平均值
        self.Small_circles=self.Small_circles[self.Small_circles[:,0]<=self.Small_circles_avg_x]


        # 画大圆(选项)
        for i in self.Big_circles:
            # 画出外圆
            cv2.circle(self.image,(i[0],i[1]),i[2],(0,255,0),3)   # 第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细
            # 画出圆心
            cv2.circle(self.image,(i[0],i[1]),2,(255,255,255),1)   # 255,255,255是白色  0,255,0是绿色
        # 画小圆(题目)
        for i in self.Small_circles:
            # 画出外圆
            cv2.circle(self.image,(i[0],i[1]),i[2],(0,255,0),3)     # 第二参数（）内是圆心坐标，第三参数是半径，第四参数（）内是颜色，第五参数是线条粗细
            # 画出圆心
        cv2.circle(self.image,(i[0],i[1]),2,(255,255,255),1)   # 255,255,255是白色

        cv2.imshow('Draw_circles',self.image)
        print("大圆(选项)的个数是：",len(self.Big_circles))
        print("小圆(题目)的个数是：",len(self.Small_circles))

        # 对ndarray类型的数据排序   小圆是选项，大圆是题目
        # print(Small_circles)
        self.Small_circles=self.Small_circles[np.lexsort(self.Small_circles[:,:2:].T)]  # 对小圆数据矩阵按第二列排序(纵坐标排序)
        Big_circles=self.Big_circles[np.lexsort(self.Big_circles[:,:2:].T)]  # 对大圆数据矩阵按第二列排序(纵坐标排序)
        return self.Small_circles,Big_circles


# 将两道题目之间的选项筛选出来，min表示最小的y坐标，max表示最大的y坐标
    def choose(self,first):
        if first == len(self.Small_circles):  # 为该页面的最后一道题目
            min = self.Small_circles[first - 1, 1]
            max = self.self.image.shape[1]
            print("筛选的纵坐标范围", self.Small_circles[first - 1, 1], "~", self.image.shape[1])  # image.shape[1]为图片的高度
        else:
            min = self.Small_circles[first - 1, 1]
            max = self.Small_circles[first, 1]
            print("筛选的纵坐标范围", self.Small_circles[first - 1, 1], "~", self.Small_circles[first, 1], ":")
        print("开始筛选")
        xuanxiang = np.uint16(np.array([[0, 0, 0]]))  # 选项信息的第一行全为0，为无效行
        biaoji = 0
        for data in self.Big_circles[:, 1]:  # 遍历大圆(选项),找出y坐标范围在(min,max)之间的选项
            if data in range(min, max):
                xuanxiang = np.append(xuanxiang, [self.Big_circles[biaoji, :]], axis=0)  # 对矩阵追加内容，axis为0表示添加行，axis为1表示追加列
            biaoji += 1  # 标识矩阵的行
        print(xuanxiang)  # 打印选项坐标半径信息
        all_x = xuanxiang[:, 0]  # 取每个大圆(选项)的x坐标
        all_y = xuanxiang[:, 1]  # 取每个大圆(选项)的y坐标
        all_r = xuanxiang[:, 2]  # 取每个大圆(选项)的y坐标
        avg_x = np.sum(all_x) / (len(xuanxiang) - 1)  # 对每个选项的x坐标求平均值
        avg_y = np.sum(all_y) / (len(xuanxiang) - 1)  # 对每个选项的y坐标求平均值
        avg_r = np.sum(all_r) / (len(xuanxiang) - 1)  # 对每个选项的r的平均值

        # 打印选项的ABCD列表
        ABCD_array = np.uint16(np.array([[0]]))
        for i in range(1, len(xuanxiang)):
            ABCD_array = np.append(ABCD_array, [[i]], axis=0)

        # print(ABCD_array,22222)
        # 如果每个选项的横坐标在误差允许范围内，近似等于平均值avg_x,(即选项在同一列)，按纵坐标y排序
        if len(all_x[np.where(all_x - avg_x <= avg_r)]) == len(xuanxiang):
            print("选项纵坐标排序")
            xuanxiang = xuanxiang[np.lexsort(xuanxiang[:, :2:].T)]  # 对纵坐标y排序

        # 如果每个选项的纵坐标在误差允许范围内，近似等于平均值avg_y，(即选项在同一行),按横坐标排序
        elif len(all_y[np.where(all_y - avg_y <= avg_r)]) == len(xuanxiang):
            print("选项横坐标排序")
            xuanxiang = xuanxiang[np.lexsort(xuanxiang[:, :1:].T)]  # 对横坐标x排序

        else:
            # 这里是对多个选项分布在不同行的操作
            store = np.uint16(np.array([[0, 0, 0]]))

            print("选项先按纵坐标排序分组，再对每个分组横坐标排序")
            k, j = 0, 1  # k用于临时存储y坐标数值,j用于标识遍历选项的时候的下标
            store_1 = np.uint16(np.array([[0, 0, 0]]))

            for select in xuanxiang[1:, :]:
                if select[1] - k > avg_r:  # 判断是否在同一行
                    k = select[1]
                    if len(store_1) > 1:
                        store_1 = store_1[np.lexsort(store_1[:, :1:].T)]  # 按x坐标排序
                        store = np.concatenate((store, store_1[1:, :]), axis=0)
                        store_1 = np.uint16(np.array([[0, 0, 0]]))

                if select[1] - k <= avg_r:
                    store_1 = np.concatenate((store_1, [select]), axis=0)
                j += 1
                if j == len(xuanxiang):
                    store_1 = store_1[np.lexsort(store_1[:, :1:].T)]  # 按x坐标排序
                    store = np.concatenate((store, store_1[1:, :]), axis=0)
                    store_1 = np.uint16(np.array([[0, 0, 0]]))
            xuanxiang = store
        xuanxiang = np.append(xuanxiang, ABCD_array, axis=1)  # 1,2,3,4分别对应A,B,C,D
        print("筛选出的选项排序后的位置（第一行无效,最后一列的1,2,3,4分别对应A,B,C,D）")
        for i in xuanxiang[1:]:  # 去掉矩阵第一行无效数据
            print(i, chr(int(i[3]) + 64))
        print("一共有{}到题目".format(len(self.Small_circles)))
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
    check_tool = check_test("D:\\Englishtest8.jpg")     # 初始化类
    # 对图像进行形态学处理，移除对象之间的小的连接，标记连通域，去除连通区域面积小的部分(即删除小块区域)
    # 过滤掉圆,找到选中矩形框
    check_tool.find_retangles()

    # 查找圆形
    check_tool.find_circles()

    # 找题号(遍历所有题目)
    # test_num = int(input("请输入题号:(如1，2，3)"))
    test_num = 3
    check_tool.choose(test_num)

    # 生成答卷
    # 与答案比对
    # 打印成绩
    cv2.waitKey(0)
    cv2.destroyAllWindows()
