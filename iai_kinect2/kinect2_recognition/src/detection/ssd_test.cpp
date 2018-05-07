#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>


using namespace std;

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Useage: " << argv[0] << " image_file" << endl;
        return 1;
    }

    std::string file(argv[1]);
    cv::Mat src = cv::imread(file, cv::IMREAD_COLOR);

    if(src.empty())
    {
        std::cout << file << " read image fialed" << std::endl;
        return 1;
    }

    cout << "Object Detection!" << endl;
    // step 1
    if(!InitSession("./model/model.ckpt", 1, src.cols, src.rows, src.channels()))
    {
        std::cout<< "init session\n";
        return -1;
    }

    std::vector<ObjInfo> obj_boxes;

    // step 2
    if(!FeedData(src))
    {
        std::cout<< "feed data\n";
        return -2;
    }
    obj_boxes.clear();

    // step 3
    if(!Detection(obj_boxes, 0.2))
    {
        std::cout<< "detection\n";
        return -3;
    }
    for(int i = 0; i < obj_boxes.size(); i++)
    {
        std::cout<< "label:" << obj_boxes[i].label << " score:" << obj_boxes[i].conf << std::endl;
        std::cout<< "box-x:" << obj_boxes[i].bbox.x
                 << ", box-y:" << obj_boxes[i].bbox.y
                 << ", box-width:" << obj_boxes[i].bbox.width
                 << ", box-height:" << obj_boxes[i].bbox.height << std::endl;

        cv::rectangle(src, obj_boxes[i].bbox, cv::Scalar(255, 0, 0), 1);
    }
    cv::imwrite("result/result.jpg", src);
    // step 4
    ReleaseSession();
    return 0;
}
