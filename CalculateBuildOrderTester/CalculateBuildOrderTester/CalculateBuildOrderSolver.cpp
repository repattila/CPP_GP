#include "stdafx.h"
#include <vector>
//#include <stdio.h>
//#include <tchar.h>
//#include <stdlib.h>
//#include <stddef.h>
#include <algorithm>

class BuildOrderSolution
{
	char **myBuildings;
	int width;
	int height;

	std::pair<size_t, size_t> nextSteps[1000];
	int nextStepsIndex = 0;
	std::pair<size_t, size_t> first1;
	std::pair<size_t, size_t> nothing;

	inline void BuildNode(const std::pair<size_t, size_t>& position)
	{
		char *value = &myBuildings[position.first][position.second];

		(*value)--; //0-3 because of mod
		*value = (*value + 4 - 1) % 4;
		(*value)++;
		if (*value == 1)
		{
			nextStepsIndex++;
			nextSteps[nextStepsIndex] = position;
		}
	}

	inline void ProcessStep(const std::pair<size_t, size_t>& step)
	{
		myBuildings[step.first][step.second] = 0;
		if (myBuildings[step.first + 1][step.second] > 0)
			BuildNode(std::pair<size_t, size_t>(step.first + 1, step.second));
		if (myBuildings[step.first][step.second + 1] > 0)
			BuildNode(std::pair<size_t, size_t>(step.first, step.second + 1));
		if (myBuildings[step.first - 1][step.second]>0)
			BuildNode(std::pair<size_t, size_t>(step.first - 1, step.second));
		if (myBuildings[step.first][step.second - 1]>0)
			BuildNode(std::pair<size_t, size_t>(step.first, step.second - 1));
	}


	inline std::pair<size_t, size_t> GetNext1(const std::pair<size_t, size_t>& coord, int width, int height)
	{
		for (size_t j = coord.second; j < height - 1; j++)
		{
			if (myBuildings[coord.first][j] == 1)
				return std::pair<size_t, size_t>(coord.first, j);
		}
		for (size_t i = coord.first + 1; i < width - 1; i++)
		{
			for (size_t j = 0; j < height - 1; j++)
			{
				if (myBuildings[i][j] == 1)
					return std::pair<size_t, size_t>(i, j);
			}
		}
		return nothing;
	}

public:
	~BuildOrderSolution()
	{
		for (size_t i = 0; i < width; i++)
		{
			delete[] myBuildings[i];
		}
		delete myBuildings;
	}

	void CalculateBuildOrder(const std::vector<std::vector<int>>& buildings, std::vector<std::pair<size_t, size_t>> & solution)
	{
		width = buildings.size() + 2;
		height = buildings[0].size() + 2;
		nothing.first = -1;
		myBuildings = new char*[width];
		for (size_t j = 0; j < width; j++)
		{
			myBuildings[j] = new char[height];
		}

		for (size_t j = 0; j < height; j++)
		{
			myBuildings[0][j] = 0;
		}

		for (size_t i = 0; i < width - 2; i++)
		{
			myBuildings[i + 1][0] = 0;
			for (size_t j = 0; j < height - 2; j++)
			{
				myBuildings[i + 1][j + 1] = buildings[i][j];
			}
			myBuildings[i + 1][height - 1] = 0;
		}
		for (size_t j = 0; j < height; j++)
		{
			myBuildings[width - 1][j] = 0;
		}

		first1 = GetNext1(std::pair<size_t, size_t>(0, 0), width, height);
		nextSteps[nextStepsIndex] = first1;
		solution.clear();
		//	solution.resize((width - 2)*(height - 2));
		int solutionIndex = 0;
		while (true)
		{
			if (nextStepsIndex == 0)
			{
				first1 = GetNext1(first1, width, height);
				if (first1.first == -1)
					break;
				nextStepsIndex++;
				nextSteps[nextStepsIndex] = first1;
			}
			auto currentCoord = nextSteps[nextStepsIndex];
			nextStepsIndex--;

			//			solution[solutionIndex] = std::pair<size_t, size_t>(currentCoord.first - 1, currentCoord.second - 1);
			solution.push_back(std::pair<size_t, size_t>(currentCoord.first - 1, currentCoord.second - 1));
			solutionIndex++;
			ProcessStep(currentCoord);
		}
		std::reverse(solution.begin(), solution.end());
	}
};


void CalculateBuildOrder(const std::vector<std::vector<int>>& buildings, std::vector<std::pair<size_t, size_t>> & solution)
{
	BuildOrderSolution solver;
	solver.CalculateBuildOrder(buildings, solution);
}