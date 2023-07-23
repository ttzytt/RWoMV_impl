#pragma once

#define NOMINMAX
#include <execinfo.h>

#include <iostream>
#define LOG(msg)                                                             \
	std::cout << "[" << __FILE__ << ", " << __FUNCTION__ << ", " << __LINE__ \
			  << "]: " << msg << std::endl;

static inline void CHECK(bool cond) {
	do {
		if (!(cond)) {
			void* callstack[128];
			int frames_cnt = backtrace(callstack, 128);

			char** func_names = backtrace_symbols(callstack, frames_cnt);

			LOG("Runtime Error.");
			exit(-1);
		}
	} while (false);
}
