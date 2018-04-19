#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>
#include <tensorflow/core/platform/env.h>

#include <algorithm>
#include <iostream>

#include "detection.h"

// define a seesion
tensorflow::Session* session;
// define a defalut graph
tensorflow::MetaGraphDef graph_def;
tensorflow::Tensor *input_images;
// output
//std::vector<tensorflow::Tensor> boxes;
//std::vector<tensorflow::Tensor> scores;
tensorflow::Status status;

int image_width;
int image_height;
int image_channels;
int batch_size;
std::string modelFile;

bool InitSession(const std::string model_file_name, const int img_batch,
                 const int img_width, const int img_height, const int img_channels)
{
    modelFile = model_file_name;
    image_width = img_width;
    image_height = img_height;
    image_channels = img_channels;
    batch_size = img_batch;

    // crate session
    status = tensorflow::NewSession(tensorflow::SessionOptions(), &session);
    if(!status.ok())
    {
        return false;
    }

    // loading the trained graph, weights and bias
    // Read in the protobuf graph we exported
    status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), modelFile + ".meta", &graph_def);
    if(!status.ok())
    {
        return false;
    }

    // Add the graph to the session
    status = session->Create(graph_def.graph_def());
    if(!status.ok())
    {
        return false;
    }

    // Read weights and biases from the checkpoint
    tensorflow::Tensor ckptTensor(tensorflow::DT_STRING, tensorflow::TensorShape());
    ckptTensor.scalar<std::string>()() = modelFile;
    status = session->Run(
            {{graph_def.saver_def().filename_tensor_name(), ckptTensor}},
            {},
            {graph_def.saver_def().restore_op_name()},
            nullptr);

    if(!status.ok())
    {
        return false;
    }

    input_images = new tensorflow::Tensor(tensorflow::DT_UINT8,
                                          tensorflow::TensorShape{batch_size, image_width, image_height, image_channels});
    return true;
}

bool FeedData(const cv::Mat &image)
{
    auto dst_img = input_images->flat<unsigned char>().data();
    std::copy_n(image.data, image_width * image_height * image_channels, dst_img);

    return true;
}

bool Detection(std::vector<cv::Rect2f> &detect_boxes,
               std::vector<float> &detect_scores, const float threshold)
{
    std::vector<tensorflow::Tensor> results;
//    std::cout<<"session run detecting."<<std::endl;
    status = session->Run({{"image_tensor", *input_images}},
                          {"detection_boxes", "detection_scores"}, {}, &results);
//    std::cout<<"session run finished."<<std::endl;
    if(!status.ok())
        return false;

    // parse return value
    detect_boxes.clear();
    detect_scores.clear();

    auto det_scores = results[1].shaped<float, 2>({batch_size, 100});
    auto det_boxes = results[0].shaped<float, 3>({batch_size, 100, 4});

    for(int i = 0; i < batch_size; i++)
    {
        int cnt = 0;
        for(int j = 0; j < 100; j++)
        {
            if(det_scores(i, j) >= threshold)
            {
                detect_scores.push_back(det_scores(i, j));
                cnt++;
            }
        }
        for(int k = 0; k < cnt; k++)
        {
            cv::Rect2f b;
            b.x = det_boxes(i, k, 1);
            b.y = det_boxes(i, k, 0);
            b.width = det_boxes(i, k, 3) - det_boxes(i,k,1);
            b.height = det_boxes(i, k, 2) - det_boxes(i,k,0);

            detect_boxes.push_back(b);
        }
    }
//    std::cout<<"object detection finished."<<std::endl;
    return true;
}

bool ReleaseSession()
{
    if(input_images==nullptr)
    {
        delete input_images;
        input_images=nullptr;
    }
    session->Close();
    delete session;
    session = nullptr;

    return true;
}
