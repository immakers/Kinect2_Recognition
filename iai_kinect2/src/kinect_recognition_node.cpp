//标准C++头文件
#include <iostream>
#include <string>

//OpenCV头文件
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>

//ROS头文件
#include <ros/ros.h>
#include <ros/spinner.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <kinect2_bridge/kinect2_definitions.h>

#include<std_msgs/Int8.h>
#include<kinova_arm_moveit_demo/targetState.h>
#include<kinova_arm_moveit_demo/targetsVector.h>

//PCL头文件
#include<pcl/io/pcd_io.h>
#include<pcl/point_cloud.h>
#include<pcl/visualization/cloud_viewer.h>

//ssd_detection头文件
#include"ssd_test/ssd_detection.h"

//相机内参
double camera_factor = 1000;
double camera_cx = 325.5;
double camera_cy = 253.5;
double camera_fx = 518.0;
double camera_fy = 519.0;

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
std::string cam_info_str = "/kinect2/qhd/camera_info";

//点云定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//点云视窗
bool show_cloud = false;

//模型路径
std::string model_path="/home/zhenglongyu/project/iai_kinect/src/iai_kinect2/kinect2_recognition/model/frozen_inference_graph.pb";

//点云显示
pcl::visualization::CloudViewer viewer("pcd viewer");
PointCloud::Ptr clouds_show(new PointCloud);
std::vector<PointCloud::Ptr> clouds;
kinova_arm_moveit_demo::targetsVector coordinate_vec;

//图像显示
std::string window_rgb_top = "/rgb_video";
std::string window_depth_top = "/depth_video";
image_transport::Publisher image_rgb_pub;
image_transport::Publisher image_depth_pub;

//机器手话题
std::string detect_target_str = "detect_target";
std::string detect_result_str = "detect_result";
ros::Publisher detect_result_pub;

//机器手采集指令宏
bool recognition_on = false;

void InitRecognition()
{
    //    ros::NodeHandle nh;
    //    nh.param("show_cloud",show_cloud,false);
    //    std::string model_path;
    //    nh.param("model_path",model_path,std::string("/home/zhenglongyu/project/iai_kinect/src/iai_kinect2/kinect2_recognition/model"));
    //    model_path+="/model.ckpt";
    // step 1
    if(!InitSession(model_path, 1, 960, 540, 3))
    {
        ROS_ERROR_STREAM("init session failed!!!the path is :"<<model_path);
        return ;
    }
    ROS_INFO_STREAM("init success");
}

//bool PointCmp(cv::Point p1, cv::Point p2, cv::Point center)
//{
//    //向量OA和向量OB的叉积
//    double det = ((p1.x - center.x) * (p2.y - center.y) - (p2.x - center.x) * (p1.y - center.y));
//    if (det < 1e-5)
//        return true;
//    if (det > 1e-5)
//        return false;
//    //向量OA和向量OB共线，以距离判断大小
//    double d1 = (p1.x - center.x) * (p1.x - center.x) + (p1.y - center.y) * (p1.y - center.y);
//    double d2 = (p2.x - center.x) * (p2.x - center.y) + (p2.y - center.y) * (p2.y - center.y);
//    return d1 > d2;
//}

//void SortQuadrilateral(cv::Point rect[], int points_count)
//{
//    //计算重心
//    cv::Point center;
//    double X = 0, Y = 0;
//    for(int i = 0;i<points_count;i++)
//    {
//        X += rect[i].x;
//        Y += rect[i].y;
//    }
//    center.x = (int)X / points_count;
//    center.y = (int)Y / points_count;

//    //冒泡排序
//    bool sort = true;
//    while(sort)
//    {
//        sort = false;
//        for (int i = 0; i < points_count - 1; i++)
//        {
//            for (int j = 0; j < points_count - i - 1; j++)
//            {
//                if (PointCmp(rect[j], rect[j + 1], center))
//                {
//                    cv::Point tmp = rect[j];
//                    rect[j] = rect[j + 1];
//                    rect[j + 1] = tmp;
//                    sort = true;
//                }
//            }
//        }
//    }
//}

void Recognition(std::vector<ObjInfo>& obj_boxes, cv::Mat src)
{
    if(src.empty())
    {
        ROS_ERROR_STREAM(" image is empty!!!");
        return ;
    }

    ROS_INFO_STREAM("Object Detection!");

//    std::vector<ObjInfo> obj_boxes;
    //    ROS_INFO_STREAM("feed data");
    // step 2
    if(!FeedData(src))
    {
        ROS_ERROR_STREAM("feed data");
        return ;
    }
        obj_boxes.clear();
    //    ROS_INFO_STREAM("detection");
    // step 3
    if(!Detection(obj_boxes, 0.2))
    {
        ROS_ERROR_STREAM("detection");
        return ;
    }
    //    ROS_INFO_STREAM("detection success");
    //    ROS_INFO_STREAM("rectangle success");
    for(size_t i = 0; i < obj_boxes.size(); i++)
    {
        ROS_INFO_STREAM("label:"<< obj_boxes[i].label<<" score:"<<obj_boxes[i].conf);
    }
    //cv::imwrite("../result/result.jpg", src);
}

void GetCloud(std::vector<ObjInfo>& rects, cv::Mat image_rgb, cv::Mat image_depth)
{
    clouds.resize(rects.size());
    for(size_t rect_num = 0;rect_num<rects.size();rect_num++)
    {
        cv::Point* rect = rects[rect_num].bbox;
        PointCloud::Ptr temp_cloud(new PointCloud);
        cv::Mat depth_masked;
        cv::Mat depth_mask = cv::Mat::zeros(image_depth.size(),CV_8UC1);
        std::vector<std::vector<cv::Point>> contour;
        std::vector<cv::Point> pts;
        pts.push_back(rect[0]);
        pts.push_back(rect[1]);
        pts.push_back(rect[2]);
        pts.push_back(rect[3]);
        contour.push_back(pts);
        cv::drawContours(depth_mask,contour,0,cv::Scalar::all(255),-1);
        image_depth.copyTo(depth_masked,depth_mask);

        for(int i = 0;i<depth_masked.rows;i++)
        {
            for(int j = 0;j<depth_masked.cols;j++)
            {
                // 获取深度图中(i,j)处的值
                ushort d = depth_masked.ptr<ushort>(i)[j];
                // d 可能没有值，若如此，跳过此点
                if (d == 0 || d!=d)
                    continue;
                // d 存在值，则向点云增加一个点
                PointT p;

                // 计算这个点的空间坐标
                p.z = double(d) / camera_factor;
                p.x = (j- camera_cx) * p.z / camera_fx;
                p.y = (i - camera_cy) * p.z / camera_fy;

                // 从rgb图像中获取它的颜色
                // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
                p.b = image_rgb.ptr<uchar>(i)[j*3];
                p.g = image_rgb.ptr<uchar>(i)[j*3+1];
                p.r = image_rgb.ptr<uchar>(i)[j*3+2];

                // 把p加入到点云中
                if(show_cloud)
                {
                    clouds_show->points.push_back( p );
                }
                temp_cloud->points.push_back(p);
            }
        }
        temp_cloud->height = 1;
        temp_cloud->width = temp_cloud->points.size();
        clouds[rect_num] = temp_cloud;
    }
    if(show_cloud)
    {
        clouds_show->height = 1;
        clouds_show->width = clouds_show->points.size();
        clouds_show->is_dense = false;
    }
}

void ShowCloud( )
{

    viewer.showCloud(clouds_show);
}

void DrawQuadrilateral(cv::Mat& image, cv::Point rect[])
{
    if(image.channels()==1)
    {
        cv::line(image,rect[0],rect[1],cv::Scalar(255));
        cv::line(image,rect[1],rect[2],cv::Scalar(255));
        cv::line(image,rect[2],rect[3],cv::Scalar(255));
        cv::line(image,rect[3],rect[0],cv::Scalar(255));
    }
    else if(image.channels()==3)
    {
        cv::line(image,rect[0],rect[1],cv::Scalar(255,0,0));
        cv::line(image,rect[1],rect[2],cv::Scalar(255,0,0));
        cv::line(image,rect[2],rect[3],cv::Scalar(255,0,0));
        cv::line(image,rect[3],rect[0],cv::Scalar(255,0,0));
    }
}

void calculate_clouds_coordinate(std::vector<ObjInfo>&Obj_Frames)
{
    coordinate_vec.targets.clear();
    coordinate_vec.targets.resize(Obj_Frames.size());
    for(size_t i = 0;i<Obj_Frames.size();i++)
    {
        PointCloud::Ptr cloud = clouds[i];
        kinova_arm_moveit_demo::targetState coordinate;
        coordinate.tag = Obj_Frames[i].label;
        coordinate.roll = 0;
        coordinate.pitch = 0;
        coordinate.yaw = 0;
        coordinate.x = 0;
        coordinate.y = 0;
        coordinate.z = 0;
        for(size_t j = 0;j<cloud->points.size();j++)
        {
            coordinate.x += cloud->points[j].x;
            coordinate.y += cloud->points[j].y;
            coordinate.z += cloud->points[j].z;
        }
        int points_size = cloud->points.size();
        coordinate.x /= points_size;
        coordinate.y /= points_size;
        coordinate.z /= points_size;
        coordinate_vec.targets[i] = coordinate;
    }
}

void RecognitionCallback(
        const sensor_msgs::ImageConstPtr image_rgb,
        const sensor_msgs::ImageConstPtr  image_depth,
        const sensor_msgs::CameraInfoConstPtr  cam_info
        )
{
    if(recognition_on==true)
    {
        ROS_INFO_STREAM("Recognition is on!!!");

        //转换ROS图像消息到opencv图像
        cv::Mat mat_image_rgb = cv_bridge::toCvShare(image_rgb)->image;
        cv::Mat mat_image_depth = cv_bridge::toCvShare(image_depth)->image;

        //识别物体
        std::vector<ObjInfo> Obj_Frames;
        Recognition(Obj_Frames, mat_image_rgb);

        //识别结果筛选
        std::vector<ObjInfo>::iterator obj_it = Obj_Frames.begin();
        while(obj_it!=Obj_Frames.end())
        {
            cv::Point* bbox = obj_it->bbox;
            //方形约束
            double l1 = sqrt((bbox[0].x-bbox[2].x)*(bbox[0].x-bbox[2].x)+(bbox[0].y-bbox[2].y)*(bbox[0].y-bbox[2].y));
            double l2 = sqrt((bbox[1].x-bbox[3].x)*(bbox[1].x-bbox[3].x)+(bbox[1].y-bbox[3].y)*(bbox[1].y-bbox[3].y));
            if(l1*l2<1e-5)
            {
                obj_it = Obj_Frames.erase(obj_it);
                continue;
            }
            if(l1/l2<0.8 || l2/l1 < 0.8)
            {
                obj_it = Obj_Frames.erase(obj_it);
                continue;
            }

            //面积约束
            double area = 0.5*l1*l2;
            double cos_alpha = (bbox[2].x-bbox[0].x)*(bbox[3].x-bbox[1].x)+(bbox[2].y-bbox[0].y)*(bbox[3].y-bbox[1].y);
            cos_alpha=cos_alpha/(l1*l2);
            area*=sqrt(1-cos_alpha*cos_alpha);

            if(area>(mat_image_rgb.rows*mat_image_rgb.cols)*0.7)
            {
                obj_it = Obj_Frames.erase(obj_it);
                continue;
            }
            obj_it++;
        }

        //相机内参提取
        camera_fx = cam_info->K[0] != 0?cam_info->K[0]:camera_fx;
        camera_fy = cam_info->K[4] != 0?cam_info->K[4]:camera_fy;
        camera_cx = cam_info->K[2] !=0?cam_info->K[2]:camera_cx;
        camera_cy = cam_info->K[5] !=0?cam_info->K[5]:camera_cy;

        //提取对应点云
        if(show_cloud)
        {
            clouds_show->points.clear();
        }
        // GetCloud(cv::Rect(round(Obj_Frame.x), round(Obj_Frame.y), round(Obj_Frame.width), round(Obj_Frame.height)), mat_image_rgb, mat_image_depth );
        GetCloud(Obj_Frames, mat_image_rgb, mat_image_depth );

        //显示点云
        if(show_cloud)
        {
            ShowCloud();
        }

        //计算点云对应物体坐标
        calculate_clouds_coordinate(Obj_Frames);

        //发送点云对应坐标
        detect_result_pub.publish(coordinate_vec);

        for(size_t i = 0;i<Obj_Frames.size();i++)
        {
            cv::Point* vertexes = Obj_Frames[i].bbox;
            for(int j = 0; j < 4; j++)
                cv::line(mat_image_rgb, vertexes[j], vertexes[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
        }
//            DrawQuadrilateral(mat_image_rgb, Obj_Frame);
//            DrawQuadrilateral(mat_image_depth, Obj_Frame);

//        ROS_INFO("the width is %d", mat_image_rgb.cols);
//        ROS_INFO("the height is %d", mat_image_rgb.rows);

        //    //转换图像到ros消息
        //    sensor_msgs::ImagePtr rgb_image_msg = cv_bridge::CvImage(std_msgs::Header(), image_rgb->encoding , image_rgb_show).toImageMsg();
        //    sensor_msgs::ImagePtr depth_image_msg = cv_bridge::CvImage(std_msgs::Header(), image_depth->encoding , image_depth_show).toImageMsg();

        //    //广播图像消息
        //    image_rgb_pub.publish(rgb_image_msg);
        //    image_depth_pub.publish(depth_image_msg);


        //显示彩色图和深度图
        try
        {
            //cv::imshow(window_rgb_top, cv_bridge::toCvShare(image_rgb)->image);
            cv::imshow("rgb_video", mat_image_rgb);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert colored images from '%s' to 'bgr8'.", image_rgb->encoding.c_str());
        }
        try
        {
            cv::imshow("depth_video", mat_image_depth);
        }
        catch(cv_bridge::Exception& e)
        {
            ROS_ERROR("Could not convert depth images from '%s' to 'bgr8'.", image_depth->encoding.c_str());
        }

    }
    else
    {
        ROS_INFO_STREAM("Recognition is off!!!");
    }

}

void RobotSignalCallback(const std_msgs::Int8::ConstPtr& msg)
{
    if(msg->data == 1)
        recognition_on = true;
    else
        recognition_on = false;
}

void ReleaseRecognition()
{
    if(show_cloud)
    {
        clouds_show->clear();
    }
    for(size_t i = 0;i<clouds.size();i++)
    {
        clouds[i]->clear();
    }
    // step 4
    ReleaseSession();
}

int main(int argc, char ** argv)
{
    //启动ROS节点并获取句柄
    ros::init(argc, argv, "kinect2_recognition");
    ros::NodeHandle nh;

    //订阅话题
    message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, image_rgb_str, 1);
    message_filters::Subscriber<sensor_msgs::Image>image_depth_sub(nh, image_depth_str, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo>cam_info_sub(nh,  cam_info_str, 1);

    //    //发布话题
    //    image_transport::ImageTransport it(nh);
    //    image_rgb_pub = it.advertise(window_rgb_top, 1);
    //    image_depth_pub = it.advertise(window_depth_top, 1);

    //    //初始化识别过程
    //    InitRecognition(window_rgb_top, window_depth_top);
    cv::namedWindow("rgb_video",CV_WINDOW_NORMAL);
    cv::namedWindow("depth_video",CV_WINDOW_NORMAL);
    cv::startWindowThread();
    //同步深度图和彩色图
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_rgb_sub, image_depth_sub, cam_info_sub, 10);
    sync.registerCallback(boost::bind(&RecognitionCallback, _1, _2, _3));

    //机器手采集信号回调
    ros::Subscriber detect_sub = nh.subscribe(detect_target_str, 1000, RobotSignalCallback);
    detect_result_pub = nh.advertise<kinova_arm_moveit_demo::targetsVector>(detect_result_str.c_str(), 1000);

    //初始化识别
    InitRecognition();

    //ros主循环
    ros::spin();
    while(ros::ok());

    //释放资源
    ReleaseRecognition();
    return 0;
}
