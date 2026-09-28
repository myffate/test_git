#include <algorithm>
#include <math.h>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);//frame_buf有700*700个vector，每个vector是一个Vector3f(RBG)
    depth_buf.resize(w * h);//同上，每个vector里是一个float，记录深度值
    //超采样,一个像素点内分为4个采样点，增加采样点大小
    //frame_buf 不变，因为最终输出图片是700*700的
    super_color_buf.resize(w * h * 4);//700*700*4个像素点，每个一个颜色值
    super_depth_buf.resize(w * h * 4);//700*700*4个深度
}

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id(); //id=0
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

//判断点是否在三角形里，通过向量叉积判断,利用叉积公式得到z分量的值，即为叉积方向
//v[0]、v[1]、v[2]为三角形顶点坐标，定义v[0]为A，v[1]为B，v[2]为C
static bool insideTriangle(float _x, float _y, const Vector3f* _v)//等价于Vector3f _v[]
{   
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    //AB X AP , BC X BP , CA X CP 方向均一致(同号)：z分量同正负，则表示点P在三角形内部
    
    Vector2f P(_x,_y); //定义点P；
    //定义向量AB AP BC BP CA CP
    Vector2f AB = _v[1].head<2>() - _v[0].head<2>();
    Vector2f AP= P -  _v[0].head<2>();
    //计算AB X AP的z分量，z = AB.x * AP.y -  AB.y * AP.x 
    float z1 = AB.x() * AP.y() - AB.y() * AP.x();
    //不做定义，硬算(BC X BP).z
    float z2 = (_v[2].x() - _v[1].x()) * (_y - _v[1].y()) - (_v[2].y() - _v[1].y()) * (_x - _v[1].x());
    //硬算(CA X CP).z = CA.x * CP.y - CA.y * CP.x 
    float z3 = (_v[0].x() - _v[2].x()) * (_y - _v[2].y()) - (_v[0].y() - _v[2].y()) * (_x - _v[2].x());

    //边上不做处理
    return (z1>=0 && z2>=0 && z3>=0);    
}

//tuple：通用结构体，可保存三个及以上的任意类型数据
//static函数具有内部链接特性，只能在定义它的源文件中使用
static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

auto to_vec4(const Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id]; //对应的6个坐标
    auto& ind = ind_buf[ind_buffer.ind_id]; //对应两组索引{{0, 1, 2},{3, 4, 5}};
    auto& col = col_buf[col_buffer.col_id]; //对应6个颜色GGB

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;

    for (auto& i : ind) //两个三角形执行两次 i=0、i=1
    {
        Triangle t; //实例化，自动初始化坐标、颜色数据为0

        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f), //齐次坐标
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division 归一化
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation 视口变换
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2; //映射顶点的深度值
        }

        //初始化三角形顶点坐标
        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>()); //初始化v[0]、v[1]、v[2]
            //t.setVertex(i, v[i].head<3>());
            //t.setVertex(i, v[i].head<3>());
        }

        //i=0时，ind:{0, 1, 2} 赋值颜色
        auto col_x = col[i[0]]; //i[0] = 0
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        //归一化颜色为0-1
        //存进t.color[0]、[1]、[2]
        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        //rasterize_triangle(t);
        //超采样
        super_rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4(); //齐次化，v.w=1
    // TODO : Find out the bounding box of current triangle.
    //找到最大、最小的x、y坐标
    float min_x = min(v[0].x(),min(v[1].x(),v[2].x()));
    float max_x = max(v[0].x(),max(v[1].x(),v[2].x()));
    float min_y = min(v[0].y(),min(v[1].y(),v[2].y()));
    float max_y = max(v[0].y(),max(v[1].y(),v[2].y()));

    // iterate through the pixel and find if the current pixel is inside the triangle
    //判断Bounding Box中的点是否在三角形内部
    for(int x = min_x; x<= max_x;x++){
        for(int y = min_y; y<=max_y;y++){
            // If so, use the following code to get the interpolated z value.
            //对三角形内的每个像素点进行深度插值：
            //根据像素距离三个顶点的远近距离，把三个顶点的深度值按权重混合一下，得到这个像素自己的深度值。
            if(insideTriangle(x+0.5f,y+0.5f,t.v)){  //像素中心坐标(采样点坐标)
                //结构化绑定语法，alpha, beta, gamma自动承接函数返回值
                //α+β+γ=1
                auto[alpha, beta, gamma] = computeBarycentric2D(x+0.5f,y+0.5f,t.v);
                float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w()); // = 1
                //w=1时，等价于：z_interpolated = alpha * z0 + beta * z1 + gamma * z2，也就是简单的线性插值。
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;

                //执行Z-Buffer深度测试
                int index = get_index(x,y); //得到某像素点的下标
                if(z_interpolated < depth_buf[index]){ //如果该点的深度值(clear函数初始化为无限大) > 该点深度插值后的值
                    //更新该点的深度值，用于下轮循环中比较前后关系
                    depth_buf[index] = z_interpolated; 
                    //set the current pixel (use the set_pixel function) to the color of the triangle 
                    //(use getColor function) if it should be painted.
                    set_pixel(Vector3f(x, y, z_interpolated), t.getColor());//插值颜色
                }
            }
        }
    }
}

//超采样
void rst::rasterizer::super_rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4(); //齐次化，v.w=1
    // TODO : Find out the bounding box of current triangle.
    //找到最大、最小的x、y坐标
    float min_x = min(v[0].x(),min(v[1].x(),v[2].x()));
    float max_x = max(v[0].x(),max(v[1].x(),v[2].x()));
    float min_y = min(v[0].y(),min(v[1].y(),v[2].y()));
    float max_y = max(v[0].y(),max(v[1].y(),v[2].y()));

    int x_min = max(0, (int)floor(min_x)); //floor函数向下取整
    int x_max = min(width - 1, (int)ceil(max_x)); //ceil函数向上取整
    int y_min = max(0, (int)floor(min_y));
    int y_max = min(height - 1, (int)ceil(max_y));

    float offset[4][2]={ //4个子采样点的偏移量,此处默认像素(x,y)是像素左下角
        {0.25f, 0.25f},  // 左下
        {0.75f, 0.25f},  // 右下
        {0.25f, 0.75f},  // 左上
        {0.75f, 0.75f}   // 右上
    };
    //判断Bounding Box中的点是否在三角形内部
    for(int x = x_min; x<= x_max; x++){
        for(int y = y_min; y<=y_max; y++){
            //当前像素的索引
            int pixel_index = get_index(x,y);
            //子像素的前序像素索引
            int sub_index = pixel_index *4;
            for(int s=0; s<4; s++){ //从左下按逆时针遍历
                float sx = x + offset[s][0];
                float sy = y + offset[s][1];
                if(insideTriangle(sx,sy,t.v)){  //如果在三角形内部

                    //计算该子采样点的深度值
                    auto[alpha, beta, gamma] = computeBarycentric2D(sx, sy, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w()); 
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;

                    //执行Z-Buffer深度测试
                    int final_index = sub_index + s;
                    if(z_interpolated < super_depth_buf[final_index]){ //如果该点的深度值(clear函数初始化为无限大) > 该点深度插值后的值
                        super_depth_buf[final_index] = z_interpolated; //更新该点的深度值
                        //得到该点的颜色值
                        super_color_buf[final_index] = t.getColor(); //一个三角形一个色
                    }
                }
            }
            //四个子像素点都遍历完后，平均颜色(只平均在内部的颜色，即无更新的黑色不参与平均)
            Vector3f avg_color(0, 0, 0);
            int covered_count = 0;
            for (int s = 0; s < 4; ++s) {
                // 如果该子采样点在当前三角形内，则已经更新了深度值
                if (super_depth_buf[sub_index + s] < std::numeric_limits<float>::infinity()) {
                    avg_color += super_color_buf[sub_index + s];
                    covered_count++;
                }
            }
            if (covered_count > 0) {
                avg_color /= (float)covered_count;
                // 画到最终帧缓冲
                set_pixel(Vector3f((float)x, (float)y, 0), avg_color);
            }
           
        }
    }

}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
        //超采样
        std::fill(super_color_buf.begin(), super_color_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
        //超采样
        std::fill(super_depth_buf.begin(), super_depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on