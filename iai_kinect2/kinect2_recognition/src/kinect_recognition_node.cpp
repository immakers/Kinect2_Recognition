//标准C++头文件
#include <iostream>
#include <string>
#include<stdio.h>

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
#include <pcl/features/integral_image_normal.h>
#include<pcl/point_types.h>
#include<pcl/features/normal_3d.h>

//ssd_detection头文件
#include"ssd_test/ssd_detection.h"

//相机内参
double camera_factor = 255/4095;
double camera_cx = 325.5;
double camera_cy = 253.5;
double camera_fx = 518.0;
double camera_fy = 519.0;

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

  if(src.channels()!=3)
  {
    ROS_ERROR_STREAM(" image is not 3 channels !!!");
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
  clouds.resize(rects.size());
  for(size_t rect_num = 0;rect_num<rects.size();rect_num++)
  {
    cv::Point* rect_points = rects[rect_num].bbox;
    cv::Rect rect = toRect(rect_points);
//    cv::Mat depth_masked;
//    cv::Mat depth_mask = cv::Mat::zeros(image_depth.size(),CV_8UC1);

//    std::vector<std::vector<cv::Point>> contour;
//    std::vector<cv::Point> pts;
//    pts.push_back(rect.);
//    pts.push_back(rect[1]);
//    pts.push_back(rect[2]);
//    pts.push_back(rect[3]);
//    contour.push_back(pts);
//    cv::drawContours(depth_mask,contour,0,cv::Scalar::all(255),-1);
//    image_depth.copyTo(depth_masked,depth_mask);

    PointCloud::Ptr temp_cloud(new PointCloud);
    temp_cloud->width = rect.width;
    temp_cloud->height = rect.height;
    temp_cloud->is_dense = false;
    temp_cloud->points.resize(temp_cloud->width*temp_cloud->height);
    for(int i = rect.x;i<rect.x+temp_cloud->width;i++)
    {
      for(int j = rect.y;j<rect.y+temp_cloud->height;j++)
      {
        // 获取深度图中(i,j)处的值
        uchar d = image_depth.ptr<uchar>(j)[i];

        // 计算这个点的空间坐标
        PointT p;

//        // d 可能没有值，若如此，跳过此点
//        if (d == 0 || d!=d)
//        {
//          p.z = 0;
//          p.x = (j- camera_cx) * p.z / camera_fx;
//          p.y = (i - camera_cy) * p.z / camera_fy;

//          // 从rgb图像中获取它的颜色
//          // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
//          p.b = image_rgb.ptr<uchar>(i)[j*3];
//          p.g = image_rgb.ptr<uchar>(i)[j*3+1];
//          p.r = image_rgb.ptr<uchar>(i)[j*3+2];
//        }
          p.z = double(d) / camera_factor;
          p.x = (i- camera_cx) * p.z / camera_fx;
          p.y = (j - camera_cy) * p.z / camera_fy;

          // 从rgb图像中获取它的颜色
          // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
          p.b = image_rgb.ptr<uchar>(j)[i*3];
          p.g = image_rgb.ptr<uchar>(j)[i*3+1];
          p.r = image_rgb.ptr<uchar>(j)[i*3+2];
//ROS_INFO_STREAM("debug!!!");
          // 把p加入到点云中
          if(show_cloud)
          {
            clouds_show->points.push_back( p );
          }

          temp_cloud->at(i-rect.x,j-rect.y) = p;
      }
    }
    //        temp_cloud->height = 1;
    //        temp_cloud->width = temp_cloud->points.size();
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

    //downsample the pointcloud by hand
    PointCloud::Ptr temp_cloud(new PointCloud);
    temp_cloud->width = ceil(0.4*cloud->width);
    temp_cloud->height = ceil(0.4*cloud->height);

    temp_cloud->is_dense = false;
    temp_cloud->points.resize(temp_cloud->width*temp_cloud->height);
    ROS_INFO_STREAM("The width and height of the reduced cloud is "<<temp_cloud->width<<" "<<temp_cloud->height);

    for(int i = 0;i<temp_cloud->width;i++)
    {
      for(int j = 0;j<temp_cloud->height;j++)
      {
        // 获取深度图中(i,j)处的值
        temp_cloud->at(i,j) = cloud->at(i+cloud->width/2-temp_cloud->width/2,j+cloud->height/2-temp_cloud->height/2);
        if(temp_cloud->at(i,j).x != temp_cloud->at(i,j).x || temp_cloud->at(i,j).y != temp_cloud->at(i,j).y || temp_cloud->at(i,j).z != temp_cloud->at(i,j).z)
        {
          if(j>0)
            temp_cloud->at(i,j) = temp_cloud->at(i,j-1);
          else
          {
            if(i>0)
              temp_cloud->at(i,j) = temp_cloud->at(i-1,j);
            else
            {
              PointT temp_point;
              temp_point.x = 0;
              temp_point.y = 0;
              temp_point.z = 0;
              temp_point.r = 255;
              temp_point.g = 0;
              temp_point.b = 0;
              temp_point.a = 0;
              temp_cloud->at(i,j)=temp_point;
            }
          }
        }
      }
    }

    coordinate.x = 0;
    coordinate.y = 0;
    coordinate.z = 0;
    int point_count = 0;
    //take the average center as grasping point
    for(int i = 0;i<temp_cloud->width;i++)
    {
      for(int j = 0;j<temp_cloud->height;j++)
      {
        // 获取深度图中(i,j)处的值
        if(temp_cloud->at(i,j).z!=temp_cloud->at(i,j).z)
          continue;
        point_count++;
        coordinate.x += temp_cloud->at(i,j).x;
        coordinate.y += temp_cloud->at(i,j).y;
        coordinate.z += temp_cloud->at(i,j).z;
      }
    }
    coordinate.x /= point_count;
    coordinate.y /= point_count;
    coordinate.z /= point_count;

    //estimate grasping direction
    pcl::Normal norm;
    norm.normal_x = 0;
    norm.normal_y = 0;
    norm.normal_z = 0;
    int norm_count = 0;

    if(temp_cloud->width<15 || temp_cloud->height<15)
    {
      norm.normal_x = 1;
      norm.normal_y = 0;
      norm.normal_z = 0;
      norm_count = 0;
    }
    else
    {
      //estimate normals
      pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
      pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
      ne.setNormalEstimationMethod (ne.AVERAGE_3D_GRADIENT);
      ne.setMaxDepthChangeFactor(0.02f);
      ne.setNormalSmoothingSize(10.0f);
      ne.setInputCloud(temp_cloud);
      ros::Time start_esti = ros::Time::now();
      ne.compute(*normals);
      ros::Time end_esti = ros::Time::now();
      ros::Duration interval = end_esti-start_esti;
      ROS_INFO_STREAM("Estimating normal vector for "<<interval.toSec()<<" s!!!");

      for(int i = 0;i<normals->width;i++)
      {
        for(int j = 0;j<normals->height;j++)
        {
          pcl::Normal norm_temp = normals->at(i,j);
          if(norm_temp.normal_x!=norm_temp.normal_x || norm_temp.normal_y != norm_temp.normal_y || norm_temp.normal_z!=norm_temp.normal_z)
            continue;
          double norm_len = sqrt(norm_temp.normal_x*norm_temp.normal_x+norm_temp.normal_y*norm_temp.normal_y+norm_temp.normal_z*norm_temp.normal_z);
          float normal[3];
          normal[0] = norm_temp.normal_x/norm_len;
          normal[1] = norm_temp.normal_y/norm_len;
          normal[2] = norm_temp.normal_z/norm_len;
          norm.normal_x += normal[0];
          norm.normal_y += normal[1];
          norm.normal_z += normal[2];
          norm_count++;
        }
      }
      norm.normal_x /= norm_count;
      norm.normal_y /= norm_count;
      norm.normal_z /= norm_count;
    }

    ROS_INFO_STREAM("normal vector is "<<norm_count<<" "<<norm.normal_x<<" "<<norm.normal_y<<" "<<norm.normal_z);

    cv::Mat vector1(3,1,CV_32FC1),vector2(3,1,CV_32FC1),vector(3,1,CV_32FC1);
    double norm_len = sqrt(norm.normal_x*norm.normal_x+norm.normal_y*norm.normal_y+norm.normal_z*norm.normal_z);
    vector1.at<float>(0,0) = norm.normal_x/norm_len;
    vector1.at<float>(1,0) = norm.normal_y/norm_len;
    vector1.at<float>(2,0) = norm.normal_z/norm_len;
    vector2.at<float>(0,0) = 1;
    vector2.at<float>(1,0) = 0;
    vector2.at<float>(2,0) = 0;
    float theta = acos(vector1.dot(vector2));
    vector = vector1.cross(vector2);
    coordinate.qx = vector.at<float>(0,0)*sin(theta/2);
    coordinate.qy = vector.at<float>(1,0)*sin(theta/2);
    coordinate.qz = vector.at<float>(2,0)*sin(theta/2);
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
      if(obj_it->label == (obj_it+1)->label)
      {
        obj_it = Obj_Frames.erase(obj_it);
        continue;
      }

      //方形约束
//      double l1 = sqrt((bbox[0].x-bbox[2].x)*(bbox[0].x-bbox[2].x)+(bbox[0].y-bbox[2].y)*(bbox[0].y-bbox[2].y));
//      double l2 = sqrt((bbox[1].x-bbox[3].x)*(bbox[1].x-bbox[3].x)+(bbox[1].y-bbox[3].y)*(bbox[1].y-bbox[3].y));
//      if(l1*l2<1e-5)
//      {
//        obj_it = Obj_Frames.erase(obj_it);
//        continue;
//      }
//      if(l1/l2<0.8 || l2/l1 < 0.8)
//      {
//        obj_it = Obj_Frames.erase(obj_it);
//        continue;
//      }

//      //面积约束
//      double area = 0.5*l1*l2;
//      double cos_alpha = (bbox[2].x-bbox[0].x)*(bbox[3].x-bbox[1].x)+(bbox[2].y-bbox[0].y)*(bbox[3].y-bbox[1].y);
//      cos_alpha=cos_alpha/(l1*l2);
//      area*=sqrt(1-cos_alpha*cos_alpha);

//      if(area>(mat_image_rgb.rows*mat_image_rgb.cols)*0.7)
//      {
//        obj_it = Obj_Frames.erase(obj_it);
//        continue;
//      }
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
    ROS_INFO_STREAM("Getting Cloud!!!");
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
  {
      simulation_on = false;
      camera_factor = 1000;
  }
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
