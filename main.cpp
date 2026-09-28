// clang-format off
#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include "rasterizer.hpp"
#include "global.hpp"
//#include "Triangle.hpp"

using namespace std;
using namespace Eigen;

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    return model;
}

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;

    view = translate*view;
    return view;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, 
                                      float zNear, float zFar)
{
    Matrix4f projection = Matrix4f::Identity();
    float theta_half_fov = eye_fov /2 * MY_PI / 180;
    float tan_t = tan(theta_half_fov);
    Matrix4f perspective;
    perspective << 1/(aspect_ratio*tan_t), 0, 0, 0,
                            0, 1/(tan_t), 0, 0,
                            0, 0, (zFar+zNear)/(zNear-zFar),(2*zFar*zNear)/(zNear-zFar),
                            0, 0, -1, 0;
    projection = perspective * projection;
    return projection;
}

int main(int argc, const char** argv)
{
    SetConsoleOutputCP(CP_UTF8);//控制台命令
    cout << "—————————START———————————" <<endl;

    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc == 2)// 不旋转，所以无angle，命令行：./Assignment2 output.png
    {
        command_line = true;
        filename = std::string(argv[1]);
    }

    rst::rasterizer r(700, 700);
    Eigen::Vector3f eye_pos = {0,0,5};

    std::vector<Eigen::Vector3f> pos //两组顶点坐标
    {{2, 0, -2},{0, 2, -2},{-2, 0, -2},
     {3.5, -1, -5},{2.5, 1.5, -5},{-1, 0.5, -5}};

    std::vector<Eigen::Vector3i> ind //两组索引
    {{0, 1, 2},
     {3, 4, 5}};

    std::vector<Eigen::Vector3f> cols //颜色
    {
    {217.0, 238.0, 185.0},
    {217.0, 238.0, 185.0},
    {217.0, 238.0, 185.0},

    {185.0, 217.0, 238.0},
    {185.0, 217.0, 238.0},
    {185.0, 217.0, 238.0}
    };

    auto pos_id = r.load_positions(pos); //pos_id = 0
    auto ind_id = r.load_indices(ind);
    auto col_id = r.load_colors(cols);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR); //上色

        cv::imwrite(filename, image);
        return 0;
    }

    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';
    }

    return 0;
}
// clang-format on