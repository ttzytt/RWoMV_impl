#include "impl.h"
#include "utils/all_utils.h"
#include "consts.h"
#include <bits/stdc++.h>
#include <filesystem>
using namespace std;

const int FRAME_CNT = 20;

FrameInfo load_frame_info(const filesystem::path &inputDir, const int &idx) {
	Buffer2D<Float3> beauty = ReadFloat3Image(
		(inputDir / ("beauty_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> normal = ReadFloat3Image(
		(inputDir / ("normal_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> position = ReadFloat3Image(
		(inputDir / ("position_" + std::to_string(idx) + ".exr")));
	Buffer2D<float> depth = ReadFloatImage(
		(inputDir / ("depth_" + std::to_string(idx) + ".exr")));

	FrameInfo frameInfo = {beauty, depth, normal, position};
	return frameInfo;
}

int main(){
	string scene_name = "cornell_box";
    filesystem::path input_dir("./test_scenes/" + scene_name + "/input");
    filesystem::path output_dir("./test_scenes/" + scene_name + "/output");
    auto&& fir_frame = load_frame_info(input_dir, 0);
    Impl algo_impl(fir_frame);
    for (int i = 1; i < FRAME_CNT; i++) {
        cout << "processing frame " << i << "/" << FRAME_CNT << "\r" << flush;
        auto&& frame = load_frame_info(input_dir, i);
		auto&& res_all = algo_impl.process_img(frame);
		for (int pass = 0; pass < HIER_LEVEL; pass++){
            auto&& pass_res = res_all[pass];
            string pprefix = "pass_" + to_string(pass) + "_";
            auto filename = [&](const string& step_name){
                return output_dir / (pprefix + step_name + "_frame" + to_string(i) + ".exr");
            };
            WriteFloat3Image(pass_res.scale_img, output_dir / filename("scale_img"));
            WriteFloat3Image(pass_res.merge_kernel_integer, output_dir / filename("merge_kernel_integer"));
            WriteFloat3Image(pass_res.merge_kernel_subpixel, output_dir / filename("merge_kernel_subpixel"));
            WriteFloat3Image(pass_res.reproject_kernel, output_dir / filename("reproject_kernel"));
            for (int j = 0; j < HIER_LEVEL; j++){
                WriteFloatImage(pass_res.dist_kernel[j], output_dir / filename("dist_kernel_svec=" + to_string(j)));
                WriteFloatImage(pass_res.blur_kernel[j], output_dir / filename("blur_kernel_svec=" + to_string(j)));
            }
        }
    }
}