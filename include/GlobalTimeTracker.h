#pragma once
#include <vector>
#include <map>
#include <string>
#include <time.h>


class GlobalTimeTracker
{
public:
	static GlobalTimeTracker& getInstance() {
		static GlobalTimeTracker obj;
		return obj;
	}

	void add(const std::string& str, float time) {
		auto it = timers.find(str);
		if (it == timers.end()) {
			timers.insert(std::make_pair(str, time));
		}
		else {
			it->second += time;
		}
	}

	void reset() {
		timers.clear();
	}

	void show() {
		for (auto& it : timers) {
			std::cout << "(" << it.first << ": " << it.second << ")     ";
		}
		std::cout << "\n";
	}

	~GlobalTimeTracker()
	{
	}

	GlobalTimeTracker(const  GlobalTimeTracker& other) = delete;
	GlobalTimeTracker& operator=(const  GlobalTimeTracker& other) = delete;
private:

	GlobalTimeTracker() {}

	std::map<std::string, float> timers;
};


class ScopeTimer
{
public:
	ScopeTimer(const std::string& _str)  
		: str(_str)
	{
		startTime = clock();
	}

	~ScopeTimer()
	{
		float totalTime = (float)(clock() - startTime) / CLOCKS_PER_SEC;
		GlobalTimeTracker::getInstance().add(str, totalTime);
	}
private:
	std::string str;
	clock_t startTime;
};

#define SCOPE_TIMER(str) ScopeTimer _scopeTimer(str)