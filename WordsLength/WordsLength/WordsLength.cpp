// WordsLength.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <string>
#include <fstream>

#define INFILE "c:\\Users\\repat\\Downloads\\result8.txt"

bool checkOverlap(const std::string & w1, const std::string & w2, int w1L, int length) {
	for (int i = length; i > 0; i--) {
		if (w1[w1L - i] != w2[length - i])
			return false;
	}

	return true;
}

int calcLongestOverlap(const std::string * const words) {
	int o1_2 = 0;
	int w1L = words[0].length();
	int w2L = words[1].length();
	int minLength = w1L < w2L ? w1L : w2L;

	for (int o = 1; o <= minLength; o++) {
		if (checkOverlap(words[0], words[1], w1L, o))
			o1_2 = o;
	}

	return o1_2;
}

int main()
{
	unsigned long sumLength = 0;

	std::ifstream inFile(INFILE);
	if (inFile.is_open()) {
		std::string words[2];
		int wordCount = 0;
		int i = 0;

		while (std::getline(inFile, words[i]))
		{
			wordCount++;

			if (i == 1) {
				unsigned int overlap = calcLongestOverlap(words);
				sumLength += (words[1].length() - overlap);

				if (overlap < words[1].length()) {
					words[0] = words[0].append(words[1].substr(overlap, words[1].length() - overlap));
					if (words[0].length() > 64) {
						words[0] = words[0].substr(words[0].length() - 64, 64);
					}
				}
			} else {
				sumLength += words[0].length();
				i = 1;
			}
		}

		inFile.close();

		std::cout << "Read " << wordCount << " words.\n";
		std::cout << "Length: " << sumLength << "\n";
	}
	else {
		std::cerr << "Couldn't open " << INFILE << " for reading.\n";
	}
}

