//标准C++头文件
#include <iostream>
#include <string>

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

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
std::string cam_info_str = "/kinect2/qhd/camera_info";

//图像保存
std::string video_save_path = "/home/zhenglongyu/data/kinect_recognition";
std::string rgb_video_path = video_save_path + "/rgb.avi";
std::string depth_video_path = video_save_path + "/depth.avi";
cv::VideoWriter depth_video_writer, rgb_video_writer;

bool cap_bool = false;

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
        //cv::imshow(window_rgb_top, cv_bridge::toCvShare(image_rgb)->image);
        cv::imshow("rgb_video", image_rgb_show);
        if(cv::waitKey(10)=='s')
        {
          cap_bool = false;
        }
        if(cv::waitKey(10)=='c')
        {
          cap_bool = true;
        }

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

//    ROS_INFO("write depth video");
//    depth_video_writer<<image_depth_show;
//    ROS_INFO("writer rgb video");
    ROS_INFO_STREAM("the capturing state is "<<cap_bool);
    if(cap_bool==true)rgb_video_writer<<image_rgb_show;

}

//void ReleaseRecognition( )
//{
////    cv::destroyAllWindows();
//}


int main(int argc, char ** argv)
{
    //启动ROS节点并获取句柄
    ros::init(argc, argv, "kinect2_caption");
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
    cv::namedWindow("rgb_video");
    cv::namedWindow("depth_video");
    cv::startWindowThread();

    if(depth_video_writer.open(depth_video_path, CV_FOURCC('M', 'J', 'P', 'G'),33,cv::Size(960, 540))==false)
    {
      ROS_ERROR("could not open depth video writer!!!");
    }

    if(rgb_video_writer.open(rgb_video_path, CV_FOURCC('M', 'J', 'P', 'G'),33,cv::Size(960,540))==false)
    {
      ROS_ERROR("could not open rgb video writer!!!");
    }

    //同步深度图和彩色图
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_rgb_sub, image_depth_sub, cam_info_sub, 10);
    sync.registerCallback(boost::bind(&RecognitionCallback, _1, _2, _3));

    //ros主循环
    ros::spin();
    while(ros::ok());
    depth_video_writer.release();
    rgb_video_writer.release();

    //释放资源
//    ReleaseRecognition();
    return 0;
}
