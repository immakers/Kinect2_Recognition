//采集模式:采集图片还是采集视频
//#define cap_video

//标准C++头文件
#include <iostream>
#include <string>
#include <stdio.h>

//OpenCV头文件
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>

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

////文件路径配置文件
//#include"KinectCaptureConfig.h"

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
std::string cam_info_str = "/kinect2/qhd/camera_info";

std::string file_save_path = "/home/zhenglongyu/data/kinect_recognition";

#ifdef cap_video
//视频保存
std::string video_save_path = file_save_path+"/video";
std::string rgb_video_path = video_save_path + "/rgb.avi";
std::string depth_video_path = video_save_path + "/depth.avi";
cv::VideoWriter depth_video_writer, rgb_video_writer;
bool cap_bool = false;
#else
//图像保存

std::string image_save_path = file_save_path + "/image";
std::string rgb_image_path = image_save_path + "/rgb";
std::string depth_image_path = image_save_path + "/depth";
int image_count = 0;
#endif

void RecognitionCallback(
        const sensor_msgs::ImageConstPtr image_rgb,
        const sensor_msgs::ImageConstPtr  image_depth,
        const sensor_msgs::CameraInfoConstPtr  cam_info
        )
{
    //转换ROS图像消息到opencv图像
    cv::Mat mat_image_rgb = cv_bridge::toCvShare(image_rgb)->image;
    cv::Mat mat_image_depth = cv_bridge::toCvShare(image_depth)->image;

    //调整图像大小以便显示
    cv::Mat image_rgb_show;
    cv::Mat image_depth_show;
    image_rgb_show = mat_image_rgb.clone();
    image_depth_show = mat_image_depth.clone();

    //    ROS_INFO("cols is %d",mat_image_rgb.cols);
    //    ROS_INFO("rows is %d",mat_image_rgb.rows);
    //显示彩色图和深度图
    try
    {     
        cv::imshow("rgb_video", image_rgb_show);
#ifdef cap_video
        if(cv::waitKey(27)=='s')
        {
            cap_bool = false;
        }
        if(cv::waitKey(27)=='c')
        {
            cap_bool = true;
        }
#else
        if(cv::waitKey(27)=='p')
        {
            char buffer_rgb[512], buffer_depth[512];
            std::sprintf(buffer_rgb, "%s/rgb_%d.jpg",rgb_image_path.c_str(), image_count);
            std::sprintf(buffer_depth, "%s/depth_%d.jpg",depth_image_path.c_str(), image_count);
            cv::imwrite(buffer_rgb,mat_image_rgb);
            cv::imwrite(buffer_depth,mat_image_depth);
            image_count++;
            ROS_INFO_STREAM("you have captured "<<image_count<<"images!!!");
        }
#endif
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not show rgb images!");
    }
    try
    {
        cv::imshow("depth_video", image_depth_show);
    }
    catch(cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not show depth images!");
    }

#ifdef cap_video
    ROS_INFO_STREAM("The capturing state is "<<cap_bool);
    if(cap_bool==true)rgb_video_writer<<image_rgb_show;
#endif
}

void ReleaseRecognition( )
{
    cv::destroyAllWindows();
}

int main(int argc, char ** argv)
{
    //启动ROS节点并获取句柄
    ros::init(argc, argv, "kinect2_caption");
    ros::NodeHandle nh;

    //订阅话题
    message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, image_rgb_str, 1);
    message_filters::Subscriber<sensor_msgs::Image>image_depth_sub(nh, image_depth_str, 1);
    message_filters::Subscriber<sensor_msgs::CameraInfo>cam_info_sub(nh,  cam_info_str, 1);

    //初始化窗口
    cv::namedWindow("rgb_video");
    cv::namedWindow("depth_video");
    cv::startWindowThread();
#ifdef cap_video
    if(depth_video_writer.open(depth_video_path, CV_FOURCC('M', 'J', 'P', 'G'),33,cv::Size(960, 540))==false)
    {
        ROS_ERROR("could not open depth video writer!!!");
    }

    if(rgb_video_writer.open(rgb_video_path, CV_FOURCC('M', 'J', 'P', 'G'),33,cv::Size(960,540))==false)
    {
        ROS_ERROR("could not open rgb video writer!!!");
    }
#endif

    //同步深度图和彩色图
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_rgb_sub, image_depth_sub, cam_info_sub, 10);
    sync.registerCallback(boost::bind(&RecognitionCallback, _1, _2, _3));

    //ros主循环
    ros::spin();
    while(ros::ok());
#ifdef cap_video
    depth_video_writer.release();
    rgb_video_writer.release();
#endif
    //释放资源
    ReleaseRecognition();
    return 0;
}
