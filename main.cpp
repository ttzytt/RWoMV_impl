#include <bits/stdc++.h>
#include <dbg.h>

#include <filesystem>

#include "consts.h"
#include "impl.h"
#include "utils/all_utils.h"
using namespace std;

const int FRAME_CNT = 20;

FrameInfo load_frame_info(const filesystem::path& inputDir, const int& idx) {
	Buffer2D<Float3> beauty = ReadFloat3Image(
		(inputDir / ("beauty_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> normal = ReadFloat3Image(
		(inputDir / ("normal_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> position = ReadFloat3Image(
		(inputDir / ("position_" + std::to_string(idx) + ".exr")));
	Buffer2D<float> depth =
		ReadFloatImage((inputDir / ("depth_" + std::to_string(idx) + ".exr")));

	FrameInfo frameInfo = {beauty, depth, normal, position};
	return frameInfo;
}

int main() {
	string scene_name = "noise_free_cbox";
	filesystem::path input_dir("/mnt/e/prog/graphics/RWoMV_impl/test_scenes/" +
							   scene_name + "/in");
	filesystem::path output_dir("/mnt/e/prog/graphics/RWoMV_impl/test_scenes/" +
								scene_name + "/out");
	auto&& fir_frame = load_frame_info(input_dir, 0);
	Impl algo_impl(fir_frame);

	auto to_buf_of_vec3 = [](const Buffer2D<Vec2>& in) {
		Buffer2D<Vec3> out = CreateBuffer2D<Vec3>(in.m_width, in.m_height);
		for (int i = 0; i < in.m_width; i++) {
			for (int j = 0; j < in.m_height; j++) {
				out(i, j) = Float3(in(i, j).x, in(i, j).y, 0);
			}
		}
		return out;
	};

	for (int i = 0; i < FRAME_CNT; i++) {
		cout << "processing frame " << i + 1 << "/" << FRAME_CNT << "\r"
			 << flush;
		auto&& frame = load_frame_info(input_dir, i);
		auto&& res_all = algo_impl.process_img(frame);
		for (int pass = 0; pass < HIER_LEVEL; pass++) {
			auto&& pass_res = res_all[pass];
			string pprefix = "hlevel_" + to_string(pass) + "_";
			auto filename = [&](const string& step_name) {
				return ("frame" + to_string(i) + "_" + pprefix + step_name +
						".exr");
			};
			if (i != 0)
				WriteFloat3Image(pass_res.scale_img,
								 output_dir / filename("scale_img"));
			if (i != 0)
				WriteFloat3Image(pass_res.merge_kernel_integer.normalize(),
								 output_dir / filename("merge_kernel_integer"));

			auto dist_after_shift =
				CreateBuffer2D<float>(pass_res.merge_kernel_integer.m_width,
									  pass_res.merge_kernel_integer.m_height);
			for (int i = 0; i < pass_res.merge_kernel_integer.m_width; i++) {
				for (int j = 0; j < pass_res.merge_kernel_integer.m_height;
					 j++) {
					dist_after_shift(i, j) =
						pass_res.merge_kernel_integer(i, j).z;
				}
			}

            if (i != 0)
                WriteFloatImage(dist_after_shift,
                                output_dir / filename("dist_after_shift"));

			// if (i != 0)
			// WriteFloat3Image(pass_res.merge_kernel_subpixel, output_dir /
			// filename("merge_kernel_subpixel"));
			if (i != 0)
				WriteFloat3Image(
					to_buf_of_vec3(pass_res.overall_shiftv.normalize()),
					output_dir / filename("overall_shiftvec"));

			WriteFloat3Image(pass_res.reproject_kernel,
							 output_dir / filename("reproject_kernel"));
			if (i != 0)
				WriteFloatImage(pass_res.final_alpha,
								output_dir / filename("final_alpha"));
			for (int j = 0; j < pass_res.dist_kernel.size() && i != 0; j++) {
				CHECK(pass_res.dist_kernel[j].m_size ==
					  pass_res.blur_kernel[j].m_size)
				WriteFloatImage(
					pass_res.dist_kernel[j],
					output_dir / filename("dist_kernel_svec=" + to_string(j)));
				WriteFloatImage(
					pass_res.blur_kernel[j],
					output_dir / filename("blur_kernel_svec=" + to_string(j)));
			}
		}
	}
}