// CalculateBuildOrderTester.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "CalculateBuildOrderSolver.h"

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#define MAX_SIZE 10
#define MAX_RUNS_SQUARE 10
#define MAX_RUNS_RECT 100

int myrandom(int i) { return std::rand() % i; }

void build(int x, int y, int width, int height, std::vector<std::vector<int>> & buildings) {
	buildings[y][x] = 1;

	int upper = y + 1;
	if (upper < height) {
		if (buildings[upper][x] == 4)
			buildings[upper][x] = 1;
		else if (buildings[upper][x] > 0)
			buildings[upper][x] += 1;
	}

	int lower = y - 1;
	if (lower >= 0) {
		if (buildings[lower][x] == 4)
			buildings[lower][x] = 1;
		else if (buildings[lower][x] > 0)
			buildings[lower][x] += 1;
	}

	int right = x + 1;
	if (right < width) {
		if (buildings[y][right] == 4)
			buildings[y][right] = 1;
		else if (buildings[y][right] > 0)
			buildings[y][right] += 1;
	}

	int left = x - 1;
	if (left >= 0) {
		if (buildings[y][left] == 4)
			buildings[y][left] = 1;
		else if (buildings[y][left] > 0)
			buildings[y][left] += 1;
	}
}

void initBuildings(std::vector<std::vector<int>> & buildings, int width, int height) {
	for (int i = 0; i < height; i++) {
		std::vector<int> row = {};

		for (int j = 0; j < width; j++) {
			row.push_back(0);
		}

		buildings.push_back(row);
	}
}

bool testResult(const std::vector<std::pair<size_t, size_t>> & solution, const std::vector<std::vector<int>> & buildings, int width, int height) {
	std::vector<std::vector<int>> buildingsBySolution;
	initBuildings(buildingsBySolution, width, height);

	for (auto coord : solution) {
		build(coord.second, coord.first, width, height, buildingsBySolution);
	}

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (buildings[i][j] != buildingsBySolution[i][j])
				return false;
		}
	}

	return true;
}

bool runTest(int width, int height) {
	std::cout << "Running test with width: " << width << " and height: "
		<< height << '\n';

	std::vector<std::vector<int>> buildings;
	initBuildings(buildings, width, height);

	std::vector<std::pair<int, int>> coords;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			coords.push_back(std::make_pair(i, j));
		}
	}
	std::random_shuffle(coords.begin(), coords.end(), myrandom);

	for (auto coord : coords) {
		build(coord.second, coord.first, width, height, buildings);
	}

	std::vector<std::pair<size_t, size_t>> solution;
	CalculateBuildOrder(buildings, solution);

	if (testResult(solution, buildings, width, height)) {
		std::cout << "Test successful." << '\n';
		std::cout << '\n';

		return true;
	} else {
		std::cout << "Test failed." << '\n';
		std::cout << '\n';

		std::string outFilename = "FailedInputs.txt";
		std::ofstream outFile;

		outFile.open(outFilename, std::ios::out | std::ios::app);
		if (outFile.is_open()) {
			outFile << "Buildings:" << '\n';
			outFile << "{\n";
			for (int i = 0; i < height; i++) {
				outFile << "{ ";
				for (int j = 0; j < width - 1; j++) {
					outFile << buildings[i][j] << ", ";
				}
				outFile << buildings[i][width - 1] << " }\n";
			}
			outFile << "}\n";

			outFile << "\n";

			outFile.close();
		}
		else {
			std::cerr << "Couldn't open " << outFilename << " for writing.\n";
		}
		
		return false;
	}
}

int main()
{
	std::random_device rd;     // only used once to initialise (seed) engine
	std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
	std::uniform_int_distribution<int> uni(1, MAX_SIZE);

	// Random squares
	for (int i = 0; i < MAX_RUNS_SQUARE; i++) {
		int length = uni(rng);

		std::cout << "Random side length: "
			<< length << '\n';

		if (!runTest(length, length))
			break;
	}
	
	// Random rectangles
	for (int i = 0; i < MAX_RUNS_RECT; i++) {
		int width = uni(rng);
		int height = uni(rng);

		std::cout << "Random width: "
			<< width << '\n';
		std::cout << "Random height: "
			<< height << '\n';

		if (!runTest(width, height))
			break;
	}
}

