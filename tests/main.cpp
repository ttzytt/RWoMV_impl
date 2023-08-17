#include "../impl.h"
#include <bits/stdc++.h>
#include <dbg.h>
using namespace std;
// test scale kernel

FrameInfo load_frame_info(const filesystem::path &inputDir, const int &idx) {
	Buffer2D<Float3> beauty = ReadFloat3Image(
		(inputDir / ("beauty_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> normal = ReadFloat3Image(
		(inputDir / ("normal_" + std::to_string(idx) + ".exr")));
	Buffer2D<Float3> position = ReadFloat3Image(
		(inputDir / ("position_" + std::to_string(idx) + ".exr")));
	Buffer2D<float> depth =
		ReadFloatImage((inputDir / ("depth_" + std::to_string(idx) + ".exr")));

	// for (int i = 0; i < beauty.m_size; i++){
	// 	if (beauty(i).x > 0.05 && beauty(i).y < 1e-3 && beauty(i).z < 1e-3){
	// 		dbg(abs(beauty(i).x - 1), beauty(i).x, beauty(i).y, beauty(i).z);
	// 		dbg(i);
	// 		beauty(i) = Vec3(0);
	// 	}
	// }

	FrameInfo frameInfo = {beauty, depth, normal, position};
	return frameInfo;
}

	string scene_name = "noise_free_cbox";
	filesystem::path input_dir("/mnt/e/prog/graphics/RWoMV_impl/test_scenes/" +
							   scene_name + "/in");
	filesystem::path output_dir("/mnt/e/prog/graphics/RWoMV_impl/tests/out");

void test_sc(){
	auto &&fir_frame = load_frame_info(input_dir, 0);
	WriteFloat3Image(fir_frame.m_beauty, output_dir / "beauty.exr");
	auto&& scaled = Impl::scale_img_ave(fir_frame.m_beauty, 4);
	WriteFloat3Image(scaled, output_dir / "scaled.exr");
	auto&& scaled_bilinear = Impl::scale_img_bilinear(fir_frame.m_beauty, 4);
	WriteFloat3Image(scaled_bilinear, output_dir / "scaled_bilinear.exr");
}

void test_quad_fit(){
	array<float, 6> target_quad = {1, 1, 0, .5, .25 ,0};
	vector<Vec3> test_pts = {
		// Vec3{0, 1, -1},
		Vec3{1, 0, -1},
		Vec3{1, 1, -1},
		Vec3{0, 0, -1},
		Vec3{0, -1, -1},
		Vec3{-1, 0, -1},
		Vec3{-1, -1, -1},
		Vec3{2, 4, -1},
		Vec3{4, 2, -1},
	};

	for (int i = 0; i < 20; i++){
		test_pts.push_back(Vec3{rand_float() * 5, rand_float() * 5, -1});
	}

	for (auto& pt : test_pts){
		pt.z = quadric_eval(target_quad, Vec2(pt.x, pt.y));
		dbg(pt.z);
		
	}

	auto&& fitted_quad = quadric_fit(test_pts, vector<float>(test_pts.size(), 1));
	dbg(fitted_quad);
	
	auto quad_func = [&](const Vec2& in){
		return quadric_eval(fitted_quad, in);
	};

	auto&& mn_pt = two_d_grad_descent(quad_func, -Vec2(100, 100), Vec2(100, 100), MERGE_KERNEL_DESC_STEP, MERGE_KERNEL_DESC_ITER);
	dbg(mn_pt);
}

int main(){
	test_quad_fit();
	return 0;
}