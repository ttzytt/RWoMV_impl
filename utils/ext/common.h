#pragma once

#define NOMINMAX
#include <execinfo.h>
#include <string>
#include <iostream>
#define LOG(msg)                                                             \
	std::cout << "[" << __FILE__ << ", " << __FUNCTION__ << ", " << __LINE__ \
			  << "]: " << msg << std::endl;

// force inline
#define FORCE_INLINE __attribute__((always_inline)) inline

#define CHECK(cond) {\
if (!(cond)) {\
	LOG("Runtime Error, condition of: " + std::string(#cond) + "failed");\
	throw std::runtime_error("Runtime Error.");\
}\
}
