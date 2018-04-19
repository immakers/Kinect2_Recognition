#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
//#include<opencv2/highgui.hpp>

#include "detection.h"

using namespace std;

int main(int argc, char *argv[])
{
    cout << "Object Detection!" << endl;
    // step 1
    if(!InitSession("./model/model.ckpt", 1, 640, 480, 3))
    {
        std::cout<< "init session\n";
        return -1;
    }

    std::vector<cv::Rect2f> det_boxes;
    std::vector<float> det_scores;

    for(int i=1; i<=7; i++)
    {
        std::string file="./image/" + std::to_string(i) + ".png";
        cv::Mat src = cv::imread(file, cv::IMREAD_COLOR);


        if(src.empty())
        {
            std::cout << file << " read fialed" << std::endl;
            continue;
        }

        std::cout << file << std::endl;

        // step 2
        if(!FeedData(src))
        {
            std::cout<< "feed data\n";
            return -2;
        }
        det_boxes.clear();
        det_scores.clear();

        // step 3
        if(!Detection(det_boxes, det_scores, 0.1))
        {
            std::cout<< "detection\n";
            return -3;
        }
        for(int i = 0; i < det_scores.size(); i++)
        {
            std::cout<< "score:"<<det_scores[i]<<std::endl;
            std::cout<< "box: x:" << det_boxes[i].x
                     << ", box: y:" << det_boxes[i].y
                     << ", box: width:" << det_boxes[i].width
                     << ", box: height:" << det_boxes[i].height << std::endl;
        }
    }
    // step 4
    ReleaseSession();
    return 0;
}
