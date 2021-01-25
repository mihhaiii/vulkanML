#pragma once
#include <vector>
#include <map>
#include <string>
#include <time.h>
#include <iostream>


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
			timers.insert({ str, { time, 1 } });
		}
		else {
			it->second.first += time;
			it->second.second += 1;
		}
	}

	void reset() {
		timers.clear();
	}

	void show() {
		for (auto& it : timers) {
			std::cout << "(" << it.first << ": " << it.second.first << ")(" << it.second.second << ")     ";
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

	std::map<std::string, std::pair<float,int>> timers;
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