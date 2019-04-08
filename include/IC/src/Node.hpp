/*********************************************************************
*
* 	Classification and Regression Tree (CART) V1.0
* 
* 	Author: Lingbo Zhang
*	11/22/2017 at Massachusetts Institute of Technology
*	Email: lingboz2015@gmail.com
* 
**********************************************************************/
#pragma once

#include<iostream>
#include<fstream>
#include<sstream>
#include<algorithm>
#include<vector>
#include<unordered_map>
#include<unordered_set>
#include<string>
#include<math.h>
#include<cstdlib>
#include<utility>

using namespace std;

typedef double data_type;

class Node{	
public:	
	Node* left;
	Node* right;
	size_t depth;
	size_t featureIndex;
	data_type splitValue;
	data_type classResult;
	vector< vector<data_type> > dataSet;
	//default constructor
	Node() 
	{
		left = nullptr;
		right = nullptr;
		depth = 0;
		featureIndex = -1;
		splitValue = 0;
		classResult = 0;
	}

	Node(size_t d)
	{
		depth = d;
		left = nullptr;
		right = nullptr;
		featureIndex = -1;
		splitValue = 0;
		classResult = 0;
	}
	~Node() 
	{
		if (left != nullptr) {
			delete left;
			left = nullptr;
		}
		if (right != nullptr) {
			delete right;
			right = nullptr;
		}
	}
	// copy constructor
	Node(const Node& other)
		:depth(other.depth),
		featureIndex(other.featureIndex),
		splitValue(other.splitValue),
		classResult(other.classResult)
	{


	}
	
	
};

