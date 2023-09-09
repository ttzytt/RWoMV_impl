#pragma once
#include "impl.h"

#include <bits/stdc++.h>
#include <dbg.h>

#include <Eigen/Dense>

#include "utils/all_utils.h"
using namespace std;

Impl::Impl(const FrameInfo &first_frame) {
	pre_frame = first_frame;
	is_first_frame = true;
	for (int i = -MAX_SHIFT_VEC_MAG; i <= MAX_SHIFT_VEC_MAG; i++) {
		for (int j = -MAX_SHIFT_VEC_MAG; j <= MAX_SHIFT_VEC_MAG; j++) {
			single_layer_shift_vecs.push_back(Ivec2(i, j));
		}
	}
}

vector<Buffer2D<float>> Impl::dist_kernel(const Buffer2D<Color> &image,
										  const Buffer2D<Color> &prev_image,
										  const Buffer2D<Vec2> &base_shiftv) {
	vector<Buffer2D<float>> ret(single_layer_shift_vecs.size());
	for (int i = 0; i < single_layer_shift_vecs.size(); i++)
		ret[i] = CreateBuffer2D<float>(image.m_width, image.m_height);
#pragma omp parallel for
	for (int sv = 0; sv < single_layer_shift_vecs.size(); sv++) {
		auto [dx, dy] = single_layer_shift_vecs[sv];
		for (int i = 0; i < image.m_width; i++) {
			for (int j = 0; j < image.m_height; j++) {
				auto [base_x, base_y] = base_shiftv(i, j);
				
				float ni = i + dx + base_x, nj = j + dy + base_y;
				if (ni < 0 || ni >= image.m_width || nj < 0 ||
					nj >= image.m_height) {
					ret[sv](i, j) = -1;
					continue;
				}
				Color cur_c = image(i, j), shift_c = prev_image(ni, nj);
				ret[sv](i, j) = DIS_FUNC(cur_c, shift_c, single_layer_shift_vecs[sv]);
			}
		}
	}
	return ret;
}

Buffer2D<Vec3> Impl::filter_kernel(const FrameInfo &frame) {
	int w = frame.m_beauty.m_width, h = frame.m_beauty.m_height;
	Buffer2D<Vec3> ret = CreateBuffer2D<Vec3>(w, h);

	auto filter_kernel = [&](int cx, int cy) {
		vector<pair<int, int>> idxs;
		pair<int, int> xrg{max(0, cx - FILT_KERNEL_RAD),
						   min(w - 1, cx + FILT_KERNEL_RAD)},
			yrg{max(0, cy - FILT_KERNEL_RAD), min(h - 1, cy + FILT_KERNEL_RAD)};
		for (int i = xrg.first; i < xrg.second; i++) {
			for (int j = yrg.first; j < yrg.second; j++) idxs.push_back({i, j});
		}
		Float3 tot_weighted_color{0};
		float tot_weight = .0;
		Float3 cent_pos = frame.m_position(cx, cy);
		Float3 cent_normal = frame.m_normal(cx, cy);
		Float3 cent_color = frame.m_beauty(cx, cy);
#pragma omp parallel for
		for (auto [x, y] : idxs) {
			Float3 cur_pos = frame.m_position(x, y);
			Float3 cur_normal = frame.m_normal(x, y);
			Float3 cur_color = frame.m_beauty(x, y);

			float pos_term =
				-SqrLength(cur_pos - cent_pos) / (2 * Sqr(FILT_KERNEL_SIG_COORD));
			float color_term =
				-SqrLength(cur_color - cent_color) / (2 * Sqr(FILT_KERNEL_SIG_COLOR));
			float norm_dot = Dot(cur_normal, cent_normal);
			float normal_term =
				-Sqr(SafeAcos(norm_dot)) / (2 * Sqr(FILT_KERNEL_SIG_NORM));
			float plane_term = 0;
			Float3 cent2cur_vec = cur_pos - cent_pos;
			if (SqrLength(cent2cur_vec) > .0) {
				Float3 cent2cur_normvec = Normalize(cent2cur_vec);
				float disp_norm_dot = Dot(cent_normal, cent2cur_normvec);
				plane_term = -Sqr(disp_norm_dot) / (2 * Sqr(FILT_KERNEL_SIG_PLANE));
			}
			float weight =
				exp(double(pos_term + color_term + normal_term + plane_term));
			tot_weighted_color += cur_color * weight;
			tot_weight += weight;
		}
		// if (tot_weight == 0.0) {
		//     return frameInfo.m_beauty(cx, cy);
		// }

		return tot_weighted_color / tot_weight;
	};

#pragma omp parallel for
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			if (frame.m_normal(x, y) == Vec3(0)) {
				ret(x, y) = Vec3(-1);
				continue;
			}
			// TODO: Joint bilateral filter
			ret(x, y) = filter_kernel(x, y);
		}
	}

	return ret;
}

Impl::vec_of_img_t<float> Impl::blur_kernel(
	const vec_of_img_t<float> &dist_output) {
	auto ret = vec_of_img_t<float>(dist_output.size());
	// if not reserve, some mem-related error will occur, not sure why is that
	int w = dist_output[0].m_width, h = dist_output[0].m_height;
	int cur_kernel_rad = round(w * BLUR_KERNEL_RAD_RATIO);
#pragma omp parallel for
	for (int i = 0; i < dist_output.size(); i++) {
		auto &cur_dis = dist_output[i];
		auto cur_ret = CreateBuffer2D<float>(w, h);
		cur_ret.Copy(cur_dis);
		// for (int x = 0; x < w; x++) {
		// 	for (int y = 0; y < h; y++) {
		// 		if (cur_dis(x, y) == -1) cur_ret(x, y) = 0;
		// 	}
		// }
		ret[i] = cur_ret.guassian_blur(BLUR_KERNEL_GUASSIAN_SIGMA, cur_kernel_rad);
		// for (int x = 0; x < w; x++){
		// 	for (int y = 0; y < h; y++){
		// 		if (cur_dis(x, y) == -1)
		// 			cur_ret(x, y) = -1;
		// 	}
		// }
	}
	return ret;
}

Buffer2D<Vec3> Impl::merge_kernel_integer(const vec_of_img_t<float> &blur_output,
										  const Buffer2D<Vec2> &base_shiftv) {
	int w = blur_output[0].m_width, h = blur_output[0].m_height;
	Buffer2D<Vec3> ret = CreateBuffer2D<Vec3>(w, h);
	fill(ret.m_buffer.get(), ret.m_buffer.get() + ret.m_size, Vec3(-1, -1, -1));
	// x, y => shift vector, z => distance
	// in each pixel find the minimum distance and corresponding shift vector
#pragma omp parallel for
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			float min_dis = 1e9;
			Vec2 min_shift(-1, -1);
			for (int sv = 0; sv < single_layer_shift_vecs.size(); sv++) {
				auto [dx, dy] =
					single_layer_shift_vecs[sv] + base_shiftv(cx, cy);
				float cur_dis = blur_output[sv](cx, cy);
				if (cur_dis == -1) continue;
				if (cur_dis < min_dis) {
					min_dis = cur_dis;
					min_shift = single_layer_shift_vecs[sv];
				}
			}
			if (min_shift.x != -1)
				ret(cx, cy) = Vec3(min_shift.x, min_shift.y, min_dis);
		}
	}
	return ret;
}

Buffer2D<Vec3> Impl::merge_kernel_subpixel(
	const vec_of_img_t<float> &blur_output,
	const Buffer2D<Vec2> &base_shiftv, 
	const Buffer2D<Vec3> &merge_int_output) {
	// form of quadric surface: z = ax^2 + by^2 + cxy + dx + ey + f
	// phi refer to [x^2, y^2, xy, x, y, 1]
	// least square method:
	// https://www.bilibili.com/video/BV1Uu411d72H/?spm_id_from=333.337.search-card.all.click&vd_source=4de003ee9a3815aedd7d0cb2c7a12d14

	int w = blur_output[0].m_width, h = blur_output[0].m_height;
	Buffer2D<Vec3> ret = CreateBuffer2D<Vec3>(w, h);
#pragma omp parallel for
	for (int cx = 0; cx < w; cx++) {
		for (int cy = 0; cy < h; cy++) {
			vector<Vec3> pts;
			vector<float> wts;
			for (int dx = -MERGE_KERNEL_RAD; dx <= MERGE_KERNEL_RAD; dx++) {
				for (int dy = -MERGE_KERNEL_RAD; dy <= MERGE_KERNEL_RAD; dy++) {
					int nx = cx + dx, ny = cy + dy;
					if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
					if (merge_int_output(nx, ny).z == -1) continue;
					pts.push_back(merge_int_output(nx, ny));
					wts.push_back(guassian(dx, dy, MERGE_KERNEL_SIGMA));
				}
			}
			if (pts.empty()) {
				ret(cx, cy) = Vec3(-1, -1, -1);
				continue;
			}

			auto quadric_params = quadric_fit(pts, wts);
			auto _quadric_eval = [&quadric_params](const Vec2 &pt) {
				return quadric_eval(quadric_params, pt);
			};	// with parameters
			Vec3 sub_pix_shiftvec_and_val = two_d_grad_descent(
				_quadric_eval, Vec2(cx, cy) - Vec2(MERGE_KERNEL_RAD),
				Vec2(cx, cy) + Vec2(MERGE_KERNEL_RAD), MERGE_KERNEL_DESC_STEP,
				MERGE_KERNEL_DESC_ITER);

			// dbg(cx, cy, sub_pix_shiftvec_and_val);
			ret(cx, cy) = sub_pix_shiftvec_and_val - Vec3(cx, cy, 0);
			// dbg(merge_int_output(cx, cy), ret(cx, cy));
		}
	}
	return ret;
}

pair<Buffer2D<Color>, Buffer2D<float>> Impl::reproject_kernel(const Buffer2D<Vec3> &shift_vec_and_dist,
									   const Buffer2D<Color> &image,
									   const Buffer2D<Color> &prev_image) {
	CHECK(shift_vec_and_dist.m_width == image.m_width &&
		  shift_vec_and_dist.m_height == image.m_height);

	Buffer2D<Color> ret = CreateBuffer2D<Color>(image.m_width, image.m_height);
	Buffer2D<float> alpha_ret = CreateBuffer2D<float>(image.m_width,
													  image.m_height);
	int w = shift_vec_and_dist.m_width, h = shift_vec_and_dist.m_height;
#pragma omp parallel for
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < h; j++) {
			int mnx = max(i - VAR_CLAMP_RAD, 0), mny = max(j - VAR_CLAMP_RAD, 0);
			int mxx = min(i + VAR_CLAMP_RAD, w - 1), mxy = min(j + VAR_CLAMP_RAD, h - 1);
			vector<Color> colors;
			for (int x = mnx; x <= mxx; x++){
				for (int y = mny; y <= mxy; y++){
					colors.push_back(image(x, y));
				}
			}
			Color ave_col = calc_ave(colors);
			Color var_col = calc_var(colors, ave_col);	
			auto cur_vdis = shift_vec_and_dist(i, j);
			float dist = cur_vdis.z;
			float m = REJECT_KAPPA * dist - REJECT_ETA;
			m = clamp(m, .0f, 1.0f);
			float alpha_p = clamp(BLEND_ALPHA * (1 - m), .0f, 1.0f);
			CHECK(alpha_p >= 0);
			if (cur_vdis.x == -1) {
				ret(i, j) = image(i, j);
				alpha_p = 0;
			}
			alpha_ret(i, j) = alpha_p;
			if (alpha_p == 0) continue;
			const Vec2 &cur_shift = Vec2(cur_vdis.x, cur_vdis.y);
			const Color &pre_acc = prev_image(i + cur_vdis.x, j + cur_vdis.y);
			
			ret(i, j) = lerp(image(i, j), pre_acc, alpha_p);

			ret(i, j) = clamp(ret(i, j), ave_col - var_col * VAR_CLAMP_COL_BOX, ave_col + var_col * VAR_CLAMP_COL_BOX);
			// auto actual_alpha = ((ret(i, j) - image(i, j) * (1 - alpha_ret(i, j))) / pre_acc);
			// dbg(actual_alpha, alpha_ret(i), image(i, j), pre_acc);
			// CHECK(actual_alpha.x == actual_alpha.y == actual_alpha.z)
			// if (alpha_ret(i, j) != actual_alpha.x)
			// 	alpha_ret(i, j) = actual_alpha.x;
		}
	}
	return {ret, alpha_ret};
}

array<Impl::BufferInOnePass, HIER_LEVEL> Impl::process_img(
	const FrameInfo &frame) {
	Impl::output_t ret;
	if (is_first_frame) {
		is_first_frame = false;
		BufferInOnePass tmp;
		tmp.is_first_frame = true;
		ret.fill(tmp);
		for (int i = 1; i < HIER_LEVEL; i++){
			ret[i].reproject_kernel = scale_img_ave(frame.m_beauty, 1.0 / pow(HIER_REDUC_FACTOR, i));
			acc_color[i] = ret[i].reproject_kernel;
		}
		ret[0].reproject_kernel = acc_color[0] = frame.m_beauty;
		pre_frame = frame;
		return ret;
	}
	auto &&filtered_img = filter_kernel(frame);
	float sc = 1.0 / pow(HIER_REDUC_FACTOR, HIER_LEVEL - 1);
	for (int i = HIER_LEVEL - 1; i >= 0; i--, sc *= HIER_REDUC_FACTOR) {
		auto &cur_pass = ret[i];
		cur_pass.scale_img = scale_img_ave(filtered_img, sc);
		Buffer2D<Vec2> base_shift;
		if (i != HIER_LEVEL - 1) {
			base_shift =
				scale_img_ave(ret[i + 1].overall_shiftv, HIER_REDUC_FACTOR);
			CHECK(base_shift.m_size == cur_pass.scale_img.m_size);
			for (int i = 0; i < base_shift.m_size; i++){
				base_shift(i) *= HIER_REDUC_FACTOR;
			}
		} else {
			base_shift = CreateBuffer2D<Vec2>(cur_pass.scale_img.m_width,
											  cur_pass.scale_img.m_height);
		}
		CHECK(base_shift.m_size == cur_pass.scale_img.m_size);
		cur_pass.dist_kernel = dist_kernel(
			cur_pass.scale_img, acc_color[i], base_shift);
		cur_pass.blur_kernel = blur_kernel(cur_pass.dist_kernel);
		cur_pass.merge_kernel_integer =
			merge_kernel_integer(cur_pass.blur_kernel, base_shift);
		if (USE_SUB_PIXEL)
			cur_pass.merge_kernel_subpixel = merge_kernel_subpixel(
			cur_pass.blur_kernel,
			base_shift,
			cur_pass.merge_kernel_integer);
		else {
			cur_pass.merge_kernel_integer = cur_pass.merge_kernel_integer.guassian_blur(2, 4);
		}

		cur_pass.overall_shiftv = base_shift;
		auto overall_with_dis = CreateBuffer2D<Vec3>(
			cur_pass.overall_shiftv.m_width, cur_pass.overall_shiftv.m_height);
		for (int j = 0; j < cur_pass.overall_shiftv.m_size; j++) {
			auto[curdx, curdy, dis] = USE_SUB_PIXEL	&& (i < HIER_REFINE_AFTER)
						 ? cur_pass.merge_kernel_subpixel(j) : cur_pass.merge_kernel_integer(j);
			cur_pass.overall_shiftv(j) += Vec2(curdx, curdy);
			auto [x, y] = cur_pass.overall_shiftv(j);
			overall_with_dis(j) = Vec3(x, y, dis);
		}

		auto[rep_img, rep_alpha] =
			reproject_kernel(overall_with_dis, cur_pass.scale_img, acc_color[i]);
		cur_pass.reproject_kernel = rep_img;
		cur_pass.final_alpha = rep_alpha;
	}

	pre_frame = frame;
	for (int i = 0; i < HIER_LEVEL; i++){
		acc_color[i] = ret[i].reproject_kernel;
	}
	return ret;
}