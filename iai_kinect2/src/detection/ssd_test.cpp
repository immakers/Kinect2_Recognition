#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "ssd_detection.h"

using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        cout << "Useage: " << argv[0] << " image_file" << " threshold"<< endl;
        return 1;
    }

    std::string file(argv[1]);
	float detect_thresh =  std::stof(argv[2]);
	
	cv::Mat dst;

	auto start_imread = std::chrono::steady_clock::now();
    cv::Mat src = cv::imread(file, cv::IMREAD_COLOR);
	std::chrono::steady_clock::time_point end_imread = std::chrono::steady_clock::now();
	cv::cvtColor(src, dst, cv::COLOR_BGR2RGB);
	std::chrono::steady_clock::time_point end_cvtimg = std::chrono::steady_clock::now();

	// time of count
	std::chrono::duration<double> diff_imread = end_imread - start_imread;
	auto diff_cvtimg = std::chrono::duration_cast<std::chrono::microseconds>(end_cvtimg - end_imread).count() / 1000.0;
	std::cout << "image read:" << diff_imread.count() / 1000.0 << "ms, convrt channels:" << diff_cvtimg << "ms" << std::endl;


    if(src.empty())
    {
        std::cout << file << " read image fialed" << std::endl;
        return 1;
    }

    cout << "Object Detection!" << endl;
    // step 1
    if(!InitSession("./model/frozen_inference_graph.pb", 1, src.cols, src.rows, src.channels()))
    {
        std::cout<< "init session\n";
        return -1;
    }

    std::vector<ObjInfo> obj_boxes;

	auto start_feed = std::chrono::steady_clock::now();
    // step 2
    if(!FeedData(dst))
    {
        std::cout<< "feed data\n";
        return -2;
    }
	std::chrono::steady_clock::time_point end_feed = std::chrono::steady_clock::now();
    obj_boxes.clear();

	std::chrono::duration<double> diff_feed = end_feed - start_feed;
	auto start_detec = std::chrono::steady_clock::now();
    // step 3
    if(!Detection(obj_boxes, detect_thresh))
    {
        std::cout<< "detection\n";
        return -3;
    }
	std::chrono::steady_clock::time_point end_detec = std::chrono::steady_clock::now();
	auto diff_detec = std::chrono::duration_cast<std::chrono::microseconds>(end_detec - start_detec).count() / 1000.0;

	std::cout << "feed data:" << diff_feed.count() / 1000.0 << "ms, detection:" << diff_detec << "ms" << std::endl;

    for(int i = 0; i < obj_boxes.size(); i++)
    {
        std::cout<< "label:" << obj_boxes[i].label << " score:" << obj_boxes[i].conf << std::endl;
		for(int j = 0; j < 4; j++)
        	cv::line(src, obj_boxes[i].bbox[j], obj_boxes[i].bbox[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);
    }
    cv::imwrite("result/result.jpg", src);
    // step 4
    ReleaseSession();
    return 0;
}
