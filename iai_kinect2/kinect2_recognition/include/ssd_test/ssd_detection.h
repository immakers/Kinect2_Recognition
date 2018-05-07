#ifndef DETECTION_H
#define DETECTION_H

//#include <opencv2/core.hpp>
#include<opencv2/core/core.hpp>

#include <vector>
#include <string>

struct ObjInfo
{
	int label;
	float conf;
	cv::Rect bbox;
	ObjInfo(){}
	ObjInfo(int classes, float confidence, cv::Rect box)
		:label(classes), conf(confidence), bbox(box)
	{}
};

/* step 1: 初始化会话
 * modelFileName: 模型文件名 e.g. model.ckpt
 * img_batch:   值为1
 * img_width:   图像宽度
 * img_height:  图像高度
 * img_channels:图像通道数
 * 失败返回 false，成功返回 true
 */
bool InitSession(const std::string model_file_name, const int img_batch,
                 const int img_width, const int img_height, const int img_channels);

/* step 2: 设置预测图片数据
 * image: 需要预测的图片
 * 失败返回 false，成功返回 true
 */
bool FeedData(const cv::Mat &image);

/* step 3: 检测
 * obj_boxes:  传出参数，每一类的label, confidence, box
 * threshold:  传入参数，置信度阈值
 * 失败返回 false，成功返回 true
 */
bool Detection(std::vector<ObjInfo> &obj_boxes, const float threshold);

/* step 4: 关闭会话，释放资源
 * 失败返回 false，成功返回 true
 */
bool ReleaseSession(void);

#endif // DETECTION_H
