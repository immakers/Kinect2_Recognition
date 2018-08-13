//标准C++头文件
#include <iostream>
#include <string>
#include<stdio.h>

//OpenCV头文件
#include<opencv2/opencv.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

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
//#include<pcl/io/pcd_io.h>
//#include<pcl/point_cloud.h>
#include<pcl/visualization/cloud_viewer.h>
//#include <pcl/features/integral_image_normal.h>
//#include<pcl/point_types.h>
//#include<pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include<pcl/common/eigen.h>
//#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/PointIndices.h>
#include<pcl/search/search.h>

//ssd_detection头文件
#include"ssd_test/ssd_detection.h"

//相机内参
double camera_factor = 1000;
double camera_cx = 479.8;
double camera_cy = 269.8;
double camera_fx = 540.7;
double camera_fy = 540.7;

//kinect相机话题
std::string image_rgb_str =  "/kinect2/qhd/image_color_rect";
std::string image_depth_str = "/kinect2/qhd/image_depth_rect";
std::string cam_info_str = "/kinect2/qhd/camera_info";
bool simulation_on;

//点云定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//点云视窗
bool show_cloud;

//模型路径
std::string model_path;
float recog_threshold;

//点云显示
pcl::visualization::CloudViewer viewer("pcd viewer");
PointCloud::Ptr clouds_show(new PointCloud);
std::vector<PointCloud::Ptr> clouds;
kinova_arm_moveit_demo::targetsVector coordinate_vec;
std::vector<int32_t*>px_py;

//图像显示
std::string window_rgb_top = "rgb_video";
std::string window_depth_top = "depth_video";
image_transport::Publisher image_rgb_pub;
image_transport::Publisher image_depth_pub;

//机器手话题
std::string detect_target_str = "detect_target";
std::string detect_result_str = "detect_result";
ros::Publisher detect_result_pub;
ros::Time timer;

//机器手采集指令宏
bool recognition_on = false;

bool comp(const ObjInfo &a, const ObjInfo &b){
    if (a.label < b.label)
        return true;
    else if (a.label == b.label  && a.conf < b.conf)
        return true;
    else                ///这里的else return false非常重要！！！！！
        return false;
}

bool compxy(const cv::Point &a, const cv::Point &b){
    if (a.x < b.x)
        return true;
    else if (a.x == b.x  && a.y < b.y)
        return true;
    else                ///这里的else return false非常重要！！！！！
        return false;
}

cv::Rect toRect(cv::Point* point)
{
  std::vector<cv::Point> points;
  for(int i = 0;i<4;i++)
    points.push_back(point[i]);
  sort(points.begin(), points.end(), compxy);
  cv::Rect rect = cv::Rect(points[0],points[3]);
  return rect;
}

void InitRecognition(int image_wid, int image_hei, int image_chan)
{
  // step 1
  if(!InitSession(model_path, 1, image_wid, image_hei, image_chan))
  {
    ROS_ERROR_STREAM("init session failed!!!the path is :"<<model_path);
    return ;
  }
  ROS_INFO_STREAM("init success");
  timer = ros::Time::now();
}

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
  cv::Mat res;
  cv::cvtColor(src, res, cv::COLOR_BGR2RGB);
  if(!FeedData(res))
  {
    ROS_ERROR_STREAM("feed data");
    return ;
  }
  obj_boxes.clear();
  //    ROS_INFO_STREAM("detection");
  // step 3
  if(!Detection(obj_boxes, recog_threshold))
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
  clouds.clear();
  px_py.clear();
  for(size_t rect_num = 0;rect_num<rects.size();rect_num++)
  {
    cv::Point* rect_points = rects[rect_num].bbox;
    cv::Rect rect = toRect(rect_points);
//    ROS_INFO_STREAM("the coordinate of the rect is "<<rect.x<<" "<<rect.y);
//    ROS_INFO_STREAM("the height of the rect is "<<rect.height<<" the width of the rect is"<<rect.width);

    PointCloud::Ptr temp_cloud(new PointCloud);
    temp_cloud->is_dense = false;
    for(int i = rect.x;i<rect.x+rect.width;i++)
    {
      for(int j = rect.y;j<rect.y+rect.height;j++)
      {
        //        if(!depth_mask.ptr<uchar>(i)[j])
        //          continue;
        // 获取深度图中(i,j)处的值
        ushort d = image_depth.ptr<ushort>(j)[i];

        // 计算这个点的空间坐标
        PointT p;
        p.z = double(d) / camera_factor;
        p.x = (i- camera_cx) * p.z / camera_fx;
        p.y = (j - camera_cy) * p.z / camera_fy;

        // 从rgb图像中获取它的颜色
        // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
        p.b = image_rgb.ptr<uchar>(j)[i*3];
        p.g = image_rgb.ptr<uchar>(j)[i*3+1];
        p.r = image_rgb.ptr<uchar>(j)[i*3+2];

        // 把p加入到点云中
        if(show_cloud)
        {
          clouds_show->points.push_back( p );
        }
        temp_cloud->points.push_back( p );
      }
    }
    temp_cloud->height = 1;
    temp_cloud->width = temp_cloud->points.size();
    clouds.push_back(temp_cloud);
    int32_t* temp_xy = new int32_t[2];
    temp_xy[0] = int32_t(rect.x+rect.width/2);temp_xy[1] = int32_t(rect.y+rect.height/2);
    px_py.push_back(temp_xy);
//    ROS_INFO_STREAM("the center point of the rect is "<<temp_xy[0]<<" "<<temp_xy[1]);
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
//  ROS_INFO_STREAM("the obj frames left is "<<Obj_Frames.size());
//  ROS_INFO_STREAM("the px_py size is "<<px_py.size()<<" "<<px_py[0][0]<<" "<<px_py[0][1]);
  coordinate_vec.targets.clear();
  coordinate_vec.targets.resize(Obj_Frames.size());
  for(size_t i = 0;i<Obj_Frames.size();i++)
  {
    PointCloud::Ptr cloud = clouds[i];
    int* temp_xy = px_py[i];
    kinova_arm_moveit_demo::targetState coordinate;
    coordinate.tag = Obj_Frames[i].label;
    coordinate.px = temp_xy[0];
    coordinate.py = temp_xy[1];

//    //cloud segmentation
//    ros::Time seg_begin = ros::Time::now();
//    std::vector<pcl::PointIndices> cluster_indices;
//    pcl::search::Search<PointT>::Ptr KdTree;
//    //set cluster size
//    pcl::EuclideanClusterExtraction<PointT> ec;
//    ec.setClusterTolerance (0.02); // 2cm
//    ec.setMinClusterSize (cloud->points.size()/300);
//    ec.setMaxClusterSize (cloud->points.size()*5/6);
//    //input cloud
//    ec.setSearchMethod (KdTree);
//    ec.setInputCloud (cloud);
//    ec.extract (cluster_indices);
//    ros::Duration seg_time = ros::Time::now()-seg_begin;
//    ROS_INFO_STREAM("segmentation time is "<<seg_time.toSec());
//    PointCloud::Ptr cloud_seg(new PointCloud);
//    for(int seg_i = 0;seg_i<cluster_indices.size()/2;seg_i++)
//    {
//        pcl::PointIndices seg_index = cluster_indices[seg_i];
//        for(size_t i = 0;i<seg_index.indices.size();i++)
//        {
//            PointT temp_p = cloud->points[seg_index.indices[i]];
//            cloud_seg->points.push_back(temp_p);
//        }
//    }

    ////利用PCA主元分析法获得点云的三个主方向，获取质心，计算协方差，获得协方差矩阵，求取协方差矩阵的特征值和特长向量，特征向量即为主方向。
    Eigen::Vector4f pcaCentroid;
    pcl::compute3DCentroid(*cloud, pcaCentroid);
    coordinate.x = pcaCentroid(0);
    coordinate.y = pcaCentroid(1);
    coordinate.z = pcaCentroid(2);
//    ROS_INFO_STREAM("calculate xyz:"<<coordinate.x<<" "<<coordinate.y<<" "<<coordinate.z);

    ros::Time axis_begin = ros::Time::now();
    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
//    ROS_INFO_STREAM("computeCovariance!");
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    ros::Time axis_end = ros::Time::now();
    ros::Duration axis_interval = axis_end-axis_begin;
//    ROS_INFO_STREAM("computing size "<<cloud->points.size()<<" for "<<axis_interval.toSec()<<"s!!!");

    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); //校正主方向间垂直
    eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
    eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));
    Eigen::Vector3f orient0 = eigenVectorsPCA.col(0);
    Eigen::Vector3f orient1 = eigenVectorsPCA.col(1);
    Eigen::Vector3f orient2 = eigenVectorsPCA.col(2);    
    float max_orient0 = 0, max_orient1 = 0, max_orient2 = 0;
    for(size_t i = 0;i<cloud->points.size();i++)
    {
      Eigen::Vector3f temp_vec;
      temp_vec(0) = cloud->points[i].x-coordinate.x;
      temp_vec(1) = cloud->points[i].y-coordinate.y;
      temp_vec(2) = cloud->points[i].z-coordinate.z;
      float temp_orient0 = abs(temp_vec.dot(orient0));
      float temp_orient1 = abs(temp_vec.dot(orient1));
      float temp_orient2 = abs(temp_vec.dot(orient2));
      max_orient0 = temp_orient0>max_orient0?temp_orient0:max_orient0;
      max_orient1 = temp_orient1>max_orient1?temp_orient1:max_orient1;
      max_orient2 = temp_orient2>max_orient2?temp_orient2:max_orient2;
    }
    Eigen::Vector3f orient;
    if(max_orient0>max_orient1 && max_orient0>max_orient2)
    {
      orient = orient0;
    }
    else
    {
      if(max_orient1>max_orient0 && max_orient1>max_orient2)
      {
        orient = orient1;
      }
      else
        orient = orient2;
    }
    Eigen::Vector3f orient_x(1,0,0);
    float theta = acos(orient.dot(orient_x));
    Eigen::Vector3f rotate_axis = orient.cross(orient_x);
    coordinate.qx = rotate_axis(0)*sin(theta/2);
    coordinate.qy = rotate_axis(1)*sin(theta/2);
    coordinate.qz = rotate_axis(2)*sin(theta/2);
    coordinate.qw = cos(theta/2);
    //put the calculated coordinate into the vector
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
    ros::Time start_esti = ros::Time::now();
    Recognition(Obj_Frames, mat_image_rgb);
    ros::Duration recog_interval = ros::Time::now()-start_esti;
    ROS_INFO_STREAM("Recognizing image for "<<recog_interval.toSec()<<" s!!!");

    //识别结果筛选
    sort(Obj_Frames.begin(), Obj_Frames.end(), comp);
    std::vector<ObjInfo>::iterator obj_it = Obj_Frames.begin();
    while(obj_it!=Obj_Frames.end())
    {
      //保留相同标签下置信度最大的物体
      if(obj_it != Obj_Frames.end()-1 && obj_it->label == (obj_it+1)->label)
      {
        obj_it = Obj_Frames.erase(obj_it);
        continue;
      }
      cv::Rect rect = toRect(obj_it->bbox);
      if(rect.area()>100000)
      {
        obj_it = Obj_Frames.erase(obj_it);
        continue;
      }
      obj_it++;
    }
//    ROS_INFO_STREAM(Obj_Frames.size()<<" images left!!!");

    //相机内参提取
//    ROS_INFO_STREAM("the K matrix is "<<cam_info->K[0]<<" "<<cam_info->K[4]<<" "<<cam_info->K[2]<<" "<<cam_info->K[5]);
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
//    ROS_INFO_STREAM("Getting Cloud!!!");
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

    //绘制识别框、对应的置信度、帧率
    for(size_t i = 0;i<Obj_Frames.size();i++)
    {
      cv::Point* vertexes = Obj_Frames[i].bbox;
      for(int j = 0; j < 4; j++)
        cv::line(mat_image_rgb, vertexes[j], vertexes[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
      char buffer[100];
      std::sprintf(buffer, "label:%d conf:%.2f", Obj_Frames[i].label, Obj_Frames[i].conf);
      cv::putText(mat_image_rgb, buffer, vertexes[0], cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
    }
    ros::Duration interval = ros::Time::now()-timer;
    double rate = 1/interval.toSec();
    timer = ros::Time::now();
    char fr_rate[100];
    std::sprintf(fr_rate, "%.4f FPS", rate);
    cv::putText(mat_image_rgb, fr_rate, cv::Point(0,mat_image_rgb.rows), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(255, 255, 255), 1);

    //显示彩色图和深度图
    try
    {
      //cv::imshow(window_rgb_top, cv_bridge::toCvShare(image_rgb)->image);
      cv::imshow(window_rgb_top, mat_image_rgb);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("Could not convert colored images from '%s' to 'bgr8'.", image_rgb->encoding.c_str());
    }
    try
    {
      cv::imshow(window_depth_top, mat_image_depth);
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

  //获取识别参数
  if(argc<2)
    model_path = "/home/zhenglongyu/project/catkin_ws/src/iai_kinect2/kinect2_recognition/model/frozen_inference_graph.pb";
  else model_path = argv[1];
  if(!nh.getParam("threshold", recog_threshold))
    recog_threshold = 0.2;
  if(!nh.getParam("show_cloud", show_cloud))
    show_cloud = false;
  if(!nh.getParam("simulation", simulation_on))
    simulation_on = false;


  //订阅话题
  message_filters::Subscriber<sensor_msgs::Image> image_rgb_sub(nh, image_rgb_str, 1);
  message_filters::Subscriber<sensor_msgs::Image>image_depth_sub(nh, image_depth_str, 1);
  message_filters::Subscriber<sensor_msgs::CameraInfo>cam_info_sub(nh,  cam_info_str, 1);

  cv::namedWindow(window_rgb_top,CV_WINDOW_NORMAL);
  cv::namedWindow(window_depth_top,CV_WINDOW_NORMAL);
  cv::startWindowThread();
  //同步深度图和彩色图
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync(image_rgb_sub, image_depth_sub, cam_info_sub, 10);
  sync.registerCallback(boost::bind(&RecognitionCallback, _1, _2, _3));

  //机器手采集信号回调
  ros::Subscriber detect_sub = nh.subscribe(detect_target_str, 1000, RobotSignalCallback);
  detect_result_pub = nh.advertise<kinova_arm_moveit_demo::targetsVector>(detect_result_str.c_str(), 1000);

  //初始化识别
  if(simulation_on)
        InitRecognition(800, 800, 3);
    else
        InitRecognition(960, 540, 3);

  //ros主循环
  ros::spin();
  while(ros::ok());

  //释放资源
  ReleaseRecognition();
  return 0;
}

