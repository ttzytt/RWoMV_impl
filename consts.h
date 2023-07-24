#include <array>
#include <cmath>
#include <functional>
#include "./utils/all_utils.h"
using namespace std;

// distance function parameters

const float LOTG_L1C_COEF = 0.02;
enum DIS_TYPES { L1, L2, LOG_L1, LOG_L1C };
using DIS_FUNC_SIG = function<float(const Color&, const Color&, const Vec2&)>;
const DIS_FUNC_SIG L1_FUNC = [](const Color& a, const Color& b,
								const Vec2& shift = Vec2()) -> float {
	return abs(a.x - b.x) + abs(a.y - b.y) + abs(a.z - b.z);
};

const DIS_FUNC_SIG L2_FUNC = [](const Color& a, const Color& b,
								const Vec2& shift = Vec2()) -> float {
	return sqrt(Sqr(a.x - b.x) + Sqr(a.y - b.y) + Sqr(a.z - b.z));
};

const DIS_FUNC_SIG LOG_L1_FUNC = [](const Color& a, const Color& b,
									const Vec2& shift = Vec2()) -> float {
	return log(2 + L1_FUNC(a, b, shift));
};

const DIS_FUNC_SIG LOG_L1C_FUNC = [](const Color& a, const Color& b,
									 const Vec2& shift) -> float {
	return log(2 + L1_FUNC(a, b, shift)) *
		   (2 - exp(-LOTG_L1C_COEF * SqrLength(shift.to3())));
};

const array<DIS_FUNC_SIG, 4> DIS_FUNCS{L1_FUNC, L2_FUNC, LOG_L1_FUNC,
									   LOG_L1C_FUNC};
const DIS_TYPES DIS_TYPE = LOG_L1C;
const auto DIS_FUNC = DIS_FUNCS[DIS_TYPE];

// kernel parameters

const int MAX_SHIFT_VEC_MAG = 2;  // s = [-2, 2]^2
const int BLUR_KERNEL_RAD = 5;
const float BLUR_KERNEL_GUASSIAN_SIGMA = 1.0;

const float MERGE_KERNEL_SIGMA = 1.0;
const int MERGE_KERNEL_RAD = 3;
const float MERGE_KERNEL_DESC_STEP = .05;
const float MERGE_KERNEL_DESC_ITER = 50;

// blending and rejection parameters
const float BLEND_ALPHA = 0.8;
const float REJECT_KAPPA = 0.2;
const float REJECT_ETA = .05;

// hierarchy parameters

const int HIER_LEVEL = 4;
const float HIER_REDUC_FACTOR = 4;
const int HIER_MAX_SHIFT_VEC_MAG = []() -> int {
	int ret = 0;
	for (int i = 0; i < HIER_LEVEL; i++) {
		ret += MAX_SHIFT_VEC_MAG * pow(HIER_REDUC_FACTOR, i);
	}
	return ret;
}();
const int HIER_REFINE_AFTER = 2; // do sub-pixel refinement for levels smaller than this 