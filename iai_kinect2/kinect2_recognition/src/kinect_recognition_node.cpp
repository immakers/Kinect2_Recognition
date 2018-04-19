//标准C++头文件
#include <iostream>
#include <string>

//OpenCV头文件
#include <opencv2/opencv.hpp>
#include<opencv2/core.hpp>

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

//PCL头文件
#include<pcl/io/pcd_io.h>
#include<pcl/point_cloud.h>
#include<pcl/visualization/cloud_viewer.h>

//相机内参
const double camera_factor = 1000;
const double camera_cx = 325.5;
const double camera_cy = 253.5;
const double camera_fx = 518.0;
const double camera_fy = 519.0;

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
std::string cam_info_str = "/kinect2/qhd/camera_info";

//点云定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

void InitRecognition(std::string window_rgb_str, std::string window_depth_str, pcl::visualization::CloudViewer &viewer, pcl::PointCloud::Ptr cloud)
{
    //初始化显示
    cv::namedWindow(window_rgb_str);
    cv::namedWindow(window_depth_str);
    cv::startWindowThread();

    //初始化点云
    cloud = new PointCloud;

    //初始化视窗
    viewer("pcd viewer");
}

void Recognition(cv::Rect2f& rect, cv::Mat image_rgb)
{
    rect.x = 0;
    rect.y = 0;
    rect.width = image_rgb.cols;
    rect.height = image_rgb.rows;
}

void GetCloud(cv::Rect rect, cv::Mat image_rgb, cv::Mat image_depth, pcl::PointCloud::Ptr cloud)
{
    for(int i = rect.y;i<rect.y+rect.height;i++)
        for(int j = rect.x;j<rect.x+rect.width;j++)
        {
            // 获取深度图中(i,j)处的值
            ushort d = image_depth.ptr<ushort>(i)[j];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
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
            cloud->points.push_back( p );
        }

    cloud->height = 1;
    cloud->width = cloud->points.size();
//    std::cout<<"point cloud size = "<<cloud->points.size()<<std::endl;
    cloud->is_dense = false;
}

void ShowCloud( pcl::visualization::CloudViewer &viewer, pcl::PointCloud::ConstPtr cloud)
{
    viewer.showCloud(cloud);
}

void RecognitionCallback(const sensor_msgs::ImageConstPtr& image_rgb, const sensor_msgs::ImageConstPtr & image_depth, const sensor_msgs::CameraInfoConstPtr & cam_info, pcl::visualization::CloudViewer &viewer, pcl::PointCloud::ConstPtr cloud)
{
    //转换ROS图像消息到opencv图像
    cv::Mat mat_image_rgb = cv_bridge::toCvShare(image_rgb)->image;
    cv::Mat mat_image_depth = cv_bridge::toCvShare(image_depth)->image;

    //识别物体
    cv::Rect2f Obj_Frame;
    Recognition(Obj_Frame, mat_image_rgb);

    //相机内参提取
    fx = cam_info->K[0] != 0?cam_info->K[0]:fx;
    fy = cam_info->K[4] != 0?cam_info->K[4]:fy;
    cx = cam_info->K[2] !=0?cam_info->K[2]:cx;
    cy = cam_info->K[5] !=0?cam_info->K[5]:cy;

    //调整图像大小以便显示
    cv::Mat image_rgb_show(640 ,480, CV_32FC1);
    cv::Mat image_depth_show(640 ,480, CV_32FC1);
    cv::resize(mat_image_rgb, image_rgb_show, image_rgb_show.size(), 0, 0, interpolation );
    cv::resize(mat_image_depth, image_depth_show, image_depth_show.size(), 0, 0, interpolation);
    float width_scale = image_rgb_show.cols/float(mat_image_rgb.cols), height_scale = image_rgb_show.rows/float(mat_image_rgb.rows);
    cv::Rect2f Obj_Frame_Show(Obj_Frame.x*width_scale, Obj_Frame.y*height_scale, Obj_Frame.width*width_scale, Obj_Frame.height*height_scale);
    cv::rectangle(image_rgb_show, cv::Rect(round(Obj_Frame_Show.x), round(Obj_Frame_Show.y), round(Obj_Frame_Show.width), round(Obj_Frame_Show.height)), cv::Scalar(255,0,0));

    //显示彩色图和深度图
    try
    {
        cv::imshow(window_rgb_str, image_rgb_show);
        //cv::imshow("colored video", cv_bridge::toCvShare(image_color, "bgr8")->image);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert colored images from '%s' to 'bgr8'.", image_rgb->encoding.c_str());
    }
    try
    {
        cv::imshow(window_depth_str, image_depth_show);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert depth images from '%s' to 'bgr8'.", image_depth->encoding.c_str());
    }

    //提取对应点云
    cloud->points.clear();
    GetCloud(cv::Rect(round(Obj_Frame.x), round(Obj_Frame.y), round(Obj_Frame.width), round(Obj_Frame.height)), mat_image_rgb, mat_image_depth, cloud );

    //显示点云
    ShowCloud(viewer, cloud);
}

void ReleaseRecognition( pcl::PointCloud::Ptr cloud)
{
    delete cloud;
    cv::destroyAllWindows();
}

int main(int argc, char ** argv)
{
    //启动ROS节点并获取句柄
    ros::init(argc, argv, "kinect_recognition");
    ros::NodeHandle nh;

    //订阅话题
    message_filters::Subscriber<sensor_msgs::Image> image_color_sub(nh, image_rgb_str, 1);
    message_filters::Subscriber<sensor_msgs::Image>image_depth_sub(nh, image_depth_str, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo>cam_info_sub(nh,  cam_info_str, 1);

    //图像显示框
    std::string window_rgb_str = "rgb video";
    std::string window_depth_str = "depth video";

    //点云视窗
    pcl::visualization::CloudViewer viewer();
    PointCloud::Ptr cloud;

    //初始化识别过程
    InitRecognition(window_rgb_str, window_depth_str, viewer, cloud);

    //同步深度图和彩色图
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_color_sub, image_depth_sub, cam_info_sub,   viewer,  cloud, 10);
    sync.registerCallback(boost::bind(&RecognitionCallback, _1, _2, _3, _4, _5));

    //ros主循环
    ros::spin();    
    while(ros::ok());

    //释放资源
    ReleaseRecognition(cloud);
    return 0;
}
