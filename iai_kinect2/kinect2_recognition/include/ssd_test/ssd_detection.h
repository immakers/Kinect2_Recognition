#ifndef DETECTION_H
#define DETECTION_H

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

struct ObjInfo
{
	int label;
	float conf;
	cv::Point bbox[4];

	ObjInfo(int classes, float confidence, cv::Point *box)
		:label(classes), conf(confidence)
	{
		bbox[0] = box[0];
		bbox[1] = box[1];
		bbox[2] = box[2];
		bbox[3] = box[3];
	}
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
