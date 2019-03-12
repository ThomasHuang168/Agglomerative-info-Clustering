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
#include<limits>
#include<cstdlib>
#include<utility>
#include"Node.hpp"

#define version        "V1.0"
#define version_date   "2017-11-25"

using namespace std;

struct CART_data{
	
       vector< vector<data_type> > trainData;
       vector< vector<data_type> > testData;
       int featureNum;
       int trainDataSize;
       int testDataSize;
};

typedef enum ENUM_TREE_TYPE {
	TREE_CLASSIFICATION = 0,
	TREE_REGRESSION
} TREE_TYPE;

typedef enum ENUM_SPLIT_RETURN {
	SPLIT_CONT = 0,
	SPLIT_STOP
} SPLIT_RETURN;

typedef enum ENUM_EVALUATION_MODE {
	EVAL_MODE_DEFAULT = 0x0, //Verbose and return accuarcy
	EVAL_MODE_SILENCE = 0x1,
	EVAL_MODE_MSE = 0x2
} EVALUATION_MODE;

struct CART_settings{
	int treeType;   // treeType 0: classification tree
	                //          1: regression tree
	data_type test_ratio;
	size_t maxDepth;
	size_t minCount;
// regression tree
	CART_settings(){
		treeType = TREE_CLASSIFICATION;
		test_ratio = 0.2;
		maxDepth = 8;
		minCount = 2;
//  regression tree
	}
};

class CART{
	
protected:
	Node* root;
// CART data
	CART_data data;	
// CART settings
	CART_settings settings;
	
public:
	CART(){}
	~CART()
	{
		//delete tree
		//	DeleteTree(root);
		delete root;
		root = nullptr;
	}
//	void Set_configurations();
	void Configure_Set_depth(int depth)
	{
		/******************************************
		*  Input:
		*     explicit:
		*     	1. depth;
		*  Output:
		*     	void;
		*  Function:
		*     Set the max depth of the decision tree;
		*******************************************/
		settings.maxDepth = depth;
	}
	int  Read_sampleFile(string sampleFile)

	{
		/******************************************
		*  Input:
		*     explicit:
		*        1. Sample file;
		*     implicit:
		*        1. CART data;
		*  Output:
		*     1. boolean value;
		*  Function:
		*     Read data from sample file;
		*     Update data:
		*     	1. trainData
		*     	2. testData
		*     	3. featureNum
		*     	4. trainDataSize
		*     	5. testDataSize
		*     Output data for checking purpose;
		******************************************/
		ifstream fin(sampleFile.c_str());
		if (fin.is_open()) 
		{
			string lineStr;
			while (getline(fin, lineStr)) 
			{
				istringstream lineStream(lineStr);
				string field;
				vector<data_type> lineVector;
				while (getline(lineStream, field, ',')) 
				{
					data_type fieldValue = stof(field);
					//
					lineVector.push_back(fieldValue);
					lineStream >> ws;
				}
				if (rand() % 10 <= settings.test_ratio * 10)
				{
					data.testData.push_back(lineVector);
				}
				else 
				{
					data.trainData.push_back(lineVector);
				}
			}
			//  update featureNum
			data.featureNum = data.trainData[0].size() - 1;
			data.trainDataSize = data.trainData.size();
			data.testDataSize = data.testData.size();
			// CheckInputs
			OutputData(data);
			return 0;
		}
		else 
		{
			cerr << "Error opening file: " << sampleFile << endl;
			return 1;
		}
		fin.close();
	}
	void Learn()
	{
		/******************************************
		*  Input:
		*     implicit:
		*     	1. root;
		*     	2. data;
		*  Output:
		*     void;
		*  Function:
		*     Build the decision tree
		*******************************************/
		//Build and initialization boolIndexArray
		//Update dataSet
		root = new Node(0);
		root->dataSet = data.trainData;
		//
		BuildTree(root);
		PrintTree(root);
	}
	data_type Evaluate(int mode = 0)
	{
		/******************************************
		*  Input:
		*     explicit:
		*     	1. mode;
		*     implicit:
		*     	1. data;
		*     	2. root;
		*  Output:
		*     void;
		*  Function:
		*     cross-validation;
		*******************************************/
		bool bSilence = mode & EVAL_MODE_SILENCE;
		bool bMSE = mode & EVAL_MODE_MSE;
		int count = 0;
		data_type sumError = 0.0;
		for (auto dataRow : data.testData) {
			Node *node;
			node = root;
			data_type classResult;
			while (node != nullptr)
			{
				int index = node->featureIndex;
				data_type splitValue = node->splitValue;
				if (-1 == index)
				{
					classResult = node->classResult;
					break;
				}
				if (dataRow[index] <= splitValue)
				{
					node = node->left;
				}
				else
				{
					node = node->right;
				}
			}
			sumError += (classResult - dataRow.back()) * (classResult - dataRow.back());
			if (fabs(classResult - dataRow.back()) <= 1e-9)
			{
				count++;
			}
		}
		data_type Accuracy = 100 * data_type(count) / data.testDataSize;
		data_type MeanSeqErr = sumError / data.featureNum;
		if (!bSilence)
		{
			cout << "Classification tree->prediction accuracy: " << Accuracy << "%" << endl;
			cout << "Classification tree->prediction MSE: " << MeanSeqErr << endl;
		}
		if (bMSE)
		{
			return MeanSeqErr;
		}
		else
		{
			return Accuracy;
		}
	}
	void Predict(vector<data_type> &dataRow)
	{
		/******************************************
		*  Input:
		*     explicit:
		*     	1. dataRow;
		*     implicit:
		*       1. root
		*  Output:
		*     void;
		*  Function:
		*     predict the classResult
		*******************************************/
		Node *node;
		node = root;
		data_type classResult;
		while (node != nullptr)
		{
			int index = node->featureIndex;
			data_type splitValue = node->splitValue;
			if (-1 == index)
			{
				classResult = node->classResult;
				break;
			}
			if (dataRow[index] <= splitValue)
			{
				node = node->left;
			}
			else
			{
				node = node->right;
			}
		}
		for (auto ele : dataRow)
		{
			cout << ele << ' ';
		}
		cout << classResult << endl;
	}
	bool Set_dataset(const vector<vector<data_type>> &dataset, vector<size_t> &indexTrainSet, size_t indexTest, bool transpose = false)
	{
		//Check
		bool bCheck = true;
		size_t numData = dataset.size();
		size_t numFeature = (*(dataset.begin())).size();
		if (transpose)
		{
			numData = numFeature;
			numFeature = dataset.size();
		}
		if (numData < 1 || numFeature < 1)
		{
			bCheck = false;
		}
		else if (indexTest >= numData)
		{
			bCheck = false;
		}
		else if (indexTrainSet.size() >= numData)
		{
			bCheck = false;
		}
		else
		{
			for (auto &indexTrain : indexTrainSet)
			{
				if (indexTrain >= numData)
				{
					bCheck = false;
				}
				if (indexTrain == indexTest)
				{
					bCheck = false;
				}
			}
		}
		//Work
		if (transpose)
		{
			for (auto &indexTrain : indexTrainSet)
			{
				vector<data_type> col_vector;
				for (auto &row : dataset)
				{
					col_vector.push_back(row[indexTrain]);
				}
				data.trainData.push_back(col_vector);
			}
			vector<data_type> col_vector;
			for (auto &row : dataset)
			{
				col_vector.push_back(row[indexTest]);
			}
			data.testData.push_back(col_vector);
		}
		else
		{
			for (auto &indexTrain : indexTrainSet)
			{
				data.trainData.push_back(dataset[indexTrain]);
			}
			data.testData.push_back(dataset[indexTest]);
		}
		data.featureNum = numFeature;
		data.testDataSize = data.testData.size();
		data.trainDataSize = data.trainData.size();
		return bCheck;
	}
protected:
	void  OutputData(CART_data &data)
	{
		/******************************************
		*  Input:
		*     explicit:
		*        1. CART data;;
		*  Output:
		*     1. void;
		*  Function:
		*  	Output the data stored in data.trainData...
		*  	for checking purpose;
		******************************************/
		ofstream fout("Check.txt", ofstream::out | ofstream::trunc);
		if (fout.is_open()) {
			int output_line_num;
			output_line_num = 10 < data.trainData.size() ? 10 : data.trainData.size();
			fout << "Number of features: " << data.featureNum << ';' << endl;
			fout << "Size of the trainData: " << data.trainDataSize << ';' << endl;
			fout << "Size of the testData: " << data.testDataSize << endl << endl;

			for (int i = 0; i < output_line_num; i++) {
				for (auto ele : data.trainData[i]) {
					fout << ele << ',';
				}
				fout << endl;
			}
		}
		fout.close();
	}
	int   BuildTree(Node *node)
	{
		/******************************************
		*  Input:
		*     explicit:
		*	1. node;
		*  Output:
		*     void;
		*  Function:
		*     Build the decision tree
		*******************************************/
		if (StopCriterion(node))
		{
			Calculate_classResult(node, settings.treeType);
			return 0;
		}
		else
		{
			int flag = Split(node);
			if (!flag)
			{
				BuildTree(node->left);
				BuildTree(node->right);
			}
		}
		return 0;
	}
	bool  StopCriterion(Node *node)
	{
		/******************************************
		*  Input:
		*     explicit:
		*     	1. node;
		*  Output:
		*  	1. boolean value: 1. true:  this is a terminal node
		*  			  2. false: this is not a terminal node
		*  Function:
		*  	Determine if whether to stop the split process:
		*  		True:  stop
		*  		False: continue
		*******************************************/
		//Criterion II: node depth
		if (node->depth >= settings.maxDepth)
			return true;
		//Criterion III: class count
		if (node->dataSet.size() <= settings.minCount)
			return true;
		return false;
	}
	void  Calculate_classResult(Node *node, int treeType) 
	{
		/**************************************************************
		*  Input:
		*     explicit:
		*     	1. node;
		*     	2. treeType;
		*  Output:
		*  	void;
		*  Function:
		*  	Calculate the leaf class result based on the treeType
		**************************************************************/
		if (TREE_CLASSIFICATION == treeType)
		{
			unordered_map<data_type, int> classTable;
			int count = 0;
			for (auto ele : node->dataSet) {
				classTable[ele.back()]++;
				if (count < classTable[ele.back()])
				{
					count = classTable[ele.back()];
					node->classResult = ele.back();
				}
			}

		}
		else
		{
			// for regression tree;
			data_type avg = 0;
			for (auto ele : node->dataSet) {
				avg += ele.back();
			}
			node->classResult = avg / node->dataSet.size();
		}
	}
	int   Split(Node *node) 
	{
		/******************************************
		*  Input:
		*     explicit:
		*     	1. node;
		*  Output:
		*     	1. Stop splitting, skip building node.left and node.right, return;
		*     	0. Continue splitting;
		*  Function:
		*  	1. Split dataSet according to the impurity function
		*  		classification tree: Gini index;
		*  		regression tree:  sum of square error;
		*  	2. Update node:
		*  		1. Update featureIndex, splitValue;
		*  		2. Generate left and right nodes;
		*******************************************/
		unordered_map<data_type, int> classSet;
		data_type score = 0;
		int splitCount, count;
		//
		if (TREE_CLASSIFICATION == settings.treeType) {
			for (auto ele : node->dataSet)
			{
				classSet[ele.back()]++;
			}
			for (auto ele : classSet)
			{
				score += ele.second * ele.second;
			}
			// Return if class pool is homogeneous
			if (classSet.size() == 1)
			{
				Calculate_classResult(node, settings.treeType);
				return SPLIT_STOP;
			}
		}
		else
		{
			for (auto ele : node->dataSet)
			{
				score += ele.back();
			}
		}
		//
		data_type Gini_index_min = 1;
		data_type variance_min = numeric_limits<data_type>::max();
		//	
		for (int fIndex = 0; fIndex < data.featureNum; fIndex++) {
			data_type splitValue;
			data_type Gini_index_tmp;
			data_type variance_min_tmp;

			sort(node->dataSet.begin(), node->dataSet.end(),
				[fIndex](vector<data_type> rowi, vector<data_type> rowj) {
				return rowi[fIndex] < rowj[fIndex];
			}
			);
			if (TREE_CLASSIFICATION == settings.treeType)
			{
				// classification tree
				vector< vector<data_type> >::iterator it;
				unordered_map<data_type, int> classSet_left, classSet_right(classSet);
				data_type scoreL = 0, scoreR = score;
				count = 1;
				for (it = node->dataSet.begin(); it != node->dataSet.end(); it++, count++)
				{
					splitValue = (*it)[fIndex];
					// Calculate GiniIndex
					data_type PL, PR, GiniL, GiniR;
					PL = (data_type)count / data_type(node->dataSet.size());
					PR = 1.0 - PL;
					classSet_left[(*it).back()]++;
					classSet_right[(*it).back()]--;
					// Calculate scoreL and scoreR
					scoreL += 2 * classSet_left[(*it).back()] - 1;
					scoreR -= (2 * classSet_right[(*it).back()] + 1);
					//		
					if ((it == node->dataSet.end() - 1) || (*(it + 1))[fIndex] != splitValue)
					{
						GiniL = PL * (1.0 - scoreL / count / count);
						if (count == node->dataSet.size())
						{
							GiniR = 0.0;
						}
						else
						{
							GiniR = PR * (1.0 - scoreR / (node->dataSet.size() - count) / (node->dataSet.size() - count));
						}
						Gini_index_tmp = GiniL + GiniR;
						//
						if (Gini_index_tmp < Gini_index_min)
						{
							Gini_index_min = Gini_index_tmp;
							node->featureIndex = fIndex;
							node->splitValue = splitValue;
							splitCount = count;
						}
					}
				}

			}
			else
			{
				// for regression tree;	
				vector< vector<data_type> >::iterator it;
				data_type sumL = 0, sumR = score;
				count = 1;
				for (it = node->dataSet.begin(); it != node->dataSet.end(); it++, count++)
				{
					splitValue = (*it)[fIndex];
					// Calculate GiniIndex
					data_type PL, PR;
					PL = (data_type)count / data_type(node->dataSet.size());
					PR = 1.0 - PL;
					// Calculate scoreL and scoreR

					sumL += (*it).back();
					sumR -= (*it).back();
					//
					if ((it == node->dataSet.end() - 1) || (*(it + 1))[fIndex] != splitValue)
					{
						data_type avgL, avgR;
						data_type squareErrorL, squareErrorR;
						avgL = sumL / count;
						avgR = sumR / (node->dataSet.size() - count);
						data_type scoreL = 0, scoreR = 0;
						for (vector< vector<data_type> >::iterator itTmp = node->dataSet.begin(); itTmp != it + 1; itTmp++)
						{
							scoreL += ((*itTmp).back() - avgL) * ((*itTmp).back() - avgL);
						}
						for (vector< vector<data_type> >::iterator itTmp = it + 1; itTmp != node->dataSet.end(); itTmp++)
						{
							scoreR += ((*itTmp).back() - avgR) * ((*itTmp).back() - avgR);
						}
						//		
						squareErrorL = PL * scoreL;
						if (count == node->dataSet.size())
						{
							squareErrorR = 0.0;
						}
						else
						{
							squareErrorR = PR * scoreR;
						}
						variance_min_tmp = squareErrorL + squareErrorR;
						//
						if (variance_min_tmp < variance_min)
						{
							variance_min = variance_min_tmp;
							node->featureIndex = fIndex;
							node->splitValue = splitValue;
							splitCount = count;
						}
					}
				}
			}
		}

		// Return if the class set could not be reduced
		if (splitCount == node->dataSet.size())
		{
			node->featureIndex = -1;
			Calculate_classResult(node, settings.treeType);
			return SPLIT_STOP;
		}

		// Split node
		node->left = new Node(node->depth + 1);
		node->right = new Node(node->depth + 1);
		for (auto ele : node->dataSet)
		{
			if (ele[node->featureIndex] <= node->splitValue)
			{
				node->left->dataSet.push_back(ele);
			}
			else
			{
				node->right->dataSet.push_back(ele);
			}
		}
		// clear dataSet	
		node->dataSet.clear();
		return SPLIT_CONT;
	}
//	int   DeleteTree(Node *node);
	int   PrintTree(Node *node) 
	{
		/******************************************
		*  Input:
		*     implicit:
		*     	1. root;
		*  Output:
		*       1. Tree is printed successfully;
		*       0. Tree is not printed successfully;
		*  Function:
		*     Print the decision tree
		*******************************************/
		if (node == nullptr)
		{
			return 0;
		}
		if (node->depth == 0)
		{
			printf("double DTreePredict (vector<double> X)\n{\n");
		}
		string s(node->depth + 1, '\t');
		if (node->featureIndex != -1)
		{
			printf("%sif (X[%d] <= %5.3f)\n%s{\n", s.c_str(), node->featureIndex + 1, node->splitValue, s.c_str());
		}
		else
		{
			printf("%sreturn %.3f\n", s.c_str(), node->classResult);
		}
		PrintTree(node->left);
		if (node->featureIndex != -1)
		{
			printf("%s}\n%selse\n%s{\n", s.c_str(), s.c_str(), s.c_str());
		}
		PrintTree(node->right);
		if (node->featureIndex != -1)
		{
			printf("%s}\n", s.c_str());
		}
		if (node->depth == 0)
		{
			printf("}\n");
		}
		return 0;
	}
};

