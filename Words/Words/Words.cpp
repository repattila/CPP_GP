// Words.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <fstream>
#include <unordered_map>

#define WORDCOUNT 100000
#define INFILE "c:\\Users\\repat\\Downloads\\words_final.txt"

//#define WORDCOUNT 50000
//#define INFILE "c:\\Users\\repat\\Downloads\\words_50000.txt"

using namespace std;

bool checkOverlap(const std::string & w1, const std::string & w2, int w1L, int length) {
	for (int i = length; i > 0; i--) {
		if (w1[w1L - i] != w2[length - i])
			return false;
	}

	return true;
}

void calcLongestOverlap(const std::string * const words, int w1Index, int w2Index, std::unordered_map<int, std::vector<std::pair<int, int>>> & overlapsByLength) {
	string w1 = words[w1Index];
	string w2 = words[w2Index];

	int o1_2 = 0;
	int o2_1 = 0;
	int w1L = w1.length();
	int w2L = w2.length();
	int minLength = w1L < w2L ? w1L : w2L;

	for (int o = 1; o <= minLength; o++) {
		if (checkOverlap(w1, w2, w1L, o))
			o1_2 = o;
	}

	if (o1_2 < minLength) {
		for (int o = 1; o <= minLength; o++) {
			if (checkOverlap(w2, w1, w2L, o))
				o2_1 = o;
		}
	}

	std::pair<int, int> wordsInOrder;
	int overlap = 0;
	if (o1_2 >= o2_1) {
		wordsInOrder = std::make_pair(w1Index, w2Index);
		overlap = o1_2;
	} else {
		wordsInOrder = std::make_pair(w2Index, w1Index);
		overlap = o2_1;
	}

	if (overlap > 0) {
		auto search = overlapsByLength.find(overlap);
		if (search != overlapsByLength.end()) {
			search->second.insert(search->second.end(), wordsInOrder);
		} else {
			std::vector<std::pair<int, int>> vec = { wordsInOrder };
			overlapsByLength.insert({ overlap, vec });
		}
	}
}

void includePair(std::unordered_map<int, std::pair<int, int>> & neighboursByWord, const std::pair<int, int> & pair) {
	auto search = neighboursByWord.find(pair.first);
	if (search != neighboursByWord.end()) {
		if (search->second.second == -1)
			search->second = std::make_pair(search->second.first, pair.second);
	} else {
		std::pair<int, int> right = std::make_pair(-1, pair.second);

		neighboursByWord.insert({ pair.first, right });
	}

	search = neighboursByWord.find(pair.second);
	if (search != neighboursByWord.end()) {
		if (search->second.first == -1)
			search->second = std::make_pair(pair.first, search->second.second);
	} else {
		std::pair<int, int> left = std::make_pair(pair.first, -1);

		neighboursByWord.insert({ pair.second, left });
	}
}

void collectResultToLeft(std::unordered_map<int, std::pair<int, int>> & neighboursByWord, int currWord, std::vector<int> & result) {
	auto search = neighboursByWord.find(currWord);
	if (search != neighboursByWord.end()) {
		result.insert(result.begin(), currWord);

		int nextWord = search->second.first;

		neighboursByWord.erase(search);

		if (nextWord != -1)
			collectResultToLeft(neighboursByWord, nextWord, result);
	}
}

void collectResultToRight(std::unordered_map<int, std::pair<int, int>> & neighboursByWord, int currWord, std::vector<int> & result) {
	auto search = neighboursByWord.find(currWord);
	if (search != neighboursByWord.end()) {
		result.insert(result.end(), currWord);

		int nextWord = search->second.second;

		neighboursByWord.erase(search);

		if (nextWord != -1)
			collectResultToRight(neighboursByWord, nextWord, result);
	}
}

int main()
{
	std::string inFilename = INFILE;
	std::string outFilename = "result.txt";
	std::ifstream inFile(inFilename);
	std::ofstream outFile;

	std::string * words = new std::string[WORDCOUNT];

	int i = 0;

	// Read the input file
	if (inFile.is_open()) {
		while (i < WORDCOUNT && std::getline(inFile, words[i]))
		{
			i++;
		}

		inFile.close();

		std::cout << "Read " << i << " words.\n";
	} else {
		std::cerr << "Couldn't open " << inFilename << " for reading.\n";
	}

	// Calculate longest overlap for every pair of words
	std::unordered_map<int, std::vector<std::pair<int, int>>> overlapsByLength = {};
	for (i = 0; i < WORDCOUNT; i++) {
		for (int j = i + 1; j < WORDCOUNT; j++) {
			calcLongestOverlap(words, i, j, overlapsByLength);
		}
	}

	std::cout << "Calculated overlaps.\n";

	std::unordered_map<int, std::pair<int, int>> neighboursByWord = {};

	// Collect longest overlap for each word
	for (i = 64; i > 0; i--) {
		auto search = overlapsByLength.find(i);
		if (search != overlapsByLength.end()) {
			for (auto wordPair : search->second) {
				//std::cout << "Overlap length: " << search->first << " words:" << wordPair.first << ", " << wordPair.second << endl;

				includePair(neighboursByWord, wordPair);
			}
		}
	}

	std::cout << "Collected longest overlaps for words.\n";

	std::vector<int> result = {};
	for (i = 64; i > 0; i--) {
		auto search = overlapsByLength.find(i);
		if (search != overlapsByLength.end()) {
			for (auto wordPair : search->second) {
				std::vector<int> resultPart = {};

				collectResultToLeft(neighboursByWord, wordPair.first, resultPart);
				collectResultToRight(neighboursByWord, wordPair.second, resultPart);

				result.insert(result.end(), resultPart.begin(), resultPart.end());
			}
		}
	}

	std::cout << "Got result.\n";

	// Write result
	outFile.open(outFilename, ios::out | ios::trunc);
	if (outFile.is_open()) {
		for (auto w : result) {
			outFile << words[w] << endl;
		}

		outFile.close();
	} else {
		std::cerr << "Couldn't open " << outFilename << " for writing.\n";
	}

	delete[] words;
	return 0;
}
