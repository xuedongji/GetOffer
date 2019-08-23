#pragma once

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <time.h>
#include <assert.h>

#include <string>
#include <vector>
#include <deque>
#include <list>
#include <stack>
#include <queue>
#include <set>
#include <unordered_set>
#include <map>
#include <unordered_map>

#include <cmath>
#include <functional>
#include <algorithm>
#include <numeric>

using namespace std;

typedef struct ListNode
{
	int m_value;
	ListNode* m_pNext;
	ListNode* m_pPre;
	ListNode(int val, ListNode* pNext = nullptr, ListNode* pPre = nullptr):
		m_value(val), m_pNext(pNext), m_pPre(pPre){}
}ListNode;

typedef struct ComplexListNode
{
	int m_value;
	ComplexListNode* m_pNext;
	ComplexListNode* m_pSibling;
	ComplexListNode(int val, ComplexListNode* pNext = nullptr, ComplexListNode* pSibling = nullptr) :
		m_value(val), m_pNext(pNext),m_pSibling(pSibling) {}
}ComplexListNode;

typedef struct BinaryTreeNode
{
	int m_value;
	BinaryTreeNode* m_pParent;
	BinaryTreeNode* m_pLeft;
	BinaryTreeNode* m_pRight;
	BinaryTreeNode(int val, BinaryTreeNode* pParent = NULL, BinaryTreeNode* pLeft = NULL, BinaryTreeNode* pRight = NULL) :
		m_value(val),m_pParent(pParent), m_pLeft(pLeft), m_pRight(pRight) 
	{}
}BinaryTreeNode;

//�����ӡ
template<typename T>
void print(T a[], int size)
{
	for (int i = 0; i < size; ++i)
	{
		cout << a[i] << " ";
	}
	cout << endl;
	return;
}

//�����ӡ
void printList(const ListNode* pHead)
{
	while (pHead != nullptr)
	{
		cout << pHead->m_value << " ";
		pHead = pHead->m_pNext;
	}
	cout << endl;
	return;
}
void printComplexList(const ComplexListNode* pHead)
{
	while (pHead != nullptr)
	{
		cout << pHead->m_value << " ";
		pHead = pHead->m_pNext;
	}
	cout << endl;
	return;
}
//�����ͷ�
void freeList(ListNode* pHead)
{
	ListNode* pTmp = nullptr;
	while (pHead!= nullptr)
	{
		pTmp = pHead;
		delete pHead;
		pHead = nullptr;
		pHead = pTmp->m_pNext;
	}
	return;
}
void freeComplexList(ComplexListNode* pHead)
{
	ComplexListNode* pTmp = nullptr;
	while (pHead != nullptr)
	{
		pTmp = pHead;
		delete pHead;
		pHead = nullptr;
		pHead = pTmp->m_pNext;
	}
	return;
}

//������ӡ
template<typename T>
void printVector(vector<T>& vec)
{
	for (vector<T>::iterator it = vec.begin(); it != vec.end(); ++it)
	{
		cout << *it << " ";
	}
	cout << endl;
	return;
}

//��������ӡ
void prePrintBinaryTree(BinaryTreeNode* root)
{
	if (root == NULL) return;
	cout << root->m_value << " ";
	prePrintBinaryTree(root->m_pLeft);
	prePrintBinaryTree(root->m_pRight);
	return;
}
void inPrintBinaryTree(BinaryTreeNode* root)
{
	if (root == NULL) return;
	inPrintBinaryTree(root->m_pLeft);
	cout << root->m_value << " ";
	inPrintBinaryTree(root->m_pRight);
	return;
}
void lastPrintBinaryTree(BinaryTreeNode* root)
{
	if (root == NULL) return;
	lastPrintBinaryTree(root->m_pLeft);
	lastPrintBinaryTree(root->m_pRight);
	cout << root->m_value << " ";
	return;
}
//���������ͷ�
void freeBinaryTree(BinaryTreeNode* root)
{
	if (root == NULL) return;
	freeBinaryTree(root->m_pLeft);
	freeBinaryTree(root->m_pRight);
	delete root;
	return;
}

//���������ھ���ȱʧֻ�ܽ�����ȣ�һ��Ϊ0.0000001
bool equal(double v1, double v2, double precision)
{
	double uPrecision = (precision < 0) ? (-1)*precision : precision;
	double dt = ((v1 < v2) ? v2 : v1) - ((v1 < v2) ? v1 : v2);
	return (dt > uPrecision) ? false : true;
}

template<typename T>
void swap(T* pA, T* pB)
{
	if (pA == nullptr || pB == nullptr)
	{
		throw "swap nullptr input";
		return;
	}
	T tmp = *pA;
	*pA = *pB;
	*pB = tmp;
	return;
}

//	�����㷨
void bubbleSort(int* pNum, int len)
{
	if (pNum == nullptr || len <= 0) return;
	for (int i = 0; i < len-1; ++i)
	{
		for (int j = i + 1; j < len; ++j)
		{
			if (pNum[i] > pNum[j])
			{
				int tmp = pNum[i];
				pNum[i] = pNum[j];
				pNum[j] = tmp;
			}
		}
	}
	return;
}

void selectSort(int* pNum, int len)
{
	if (pNum == nullptr || len <= 0) return;
	for (int i = 0; i < len - 1; ++i)
	{
		int minNum = INT_MAX;
		int pos = -1;
		for (int j = i; j < len; ++j)
		{
			if (pNum[j] < minNum)
			{
				minNum = pNum[j];
				pos = j;
			}
		}
		pNum[pos] = pNum[i];
		pNum[i] = minNum;
	}
	return;
}

void quickSort(int* pNum, int left,int right)
{
	if (pNum == nullptr || (right-left) <= 1) return;
	int base = pNum[left];
	int orgLeft = left;
	int orgRight = right;
	while (left < right)
	{
		while (pNum[right] > base && left < right) --right;
		if (left < right)
		{
			pNum[left] = pNum[right];
			++left;
		}

		while (pNum[left] < base && left < right) ++left;
		if (left < right)
		{
			pNum[right] = pNum[left];
			--right;
		}
	}
	pNum[left] = base;
	quickSort(pNum, orgLeft, left - 1);
	quickSort(pNum, left +1, orgRight);

	return;
}

void merge(int* pNum, int s1, int e1, int s2, int e2)
{
	int* pTmp = new int[e2 - s1 + 1];
	memset(pTmp, 0, sizeof(int)*(e2 - s1 + 1));
	int u = s1;
	int v = s2;
	int i = 0;
	while ((u <= e1) && (v <= e2))
	{
		if (pNum[u] < pNum[v]) pTmp[i++] = pNum[u++];
		else pTmp[i++] = pNum[v++];
	}
	if (u > e1)
	{
		while (v <= e2)
		{
			pTmp[i++] = pNum[v++];
		}
	}
	else
	{
		while (u <= e1)
		{
			pTmp[i++] = pNum[u++];
		}
	}
	print(pTmp, e2 - s1 + 1);
	memcpy(pNum+s1, pTmp, sizeof(int)*(e2 - s1 + 1));
	delete[] pTmp;
	return;
}
void mergeSort(int* pNum,int start,int end)
{
	if (pNum == nullptr || end <= start) return;
	int mid = (end + start)>>1;
	mergeSort(pNum, start, mid);
	mergeSort(pNum, mid+1,end);
	merge(pNum, start, mid, mid + 1, end);
	return;
}

void adjust(int* pNum, int len, int idx)
{
	int left = 2 * idx + 1;
	int right = 2 * idx + 2;
	int max = idx;
	if (left < len && pNum[left] > pNum[max]) max = left;
	if (right < len && pNum[right] > pNum[max]) max = right;
	if (max != idx)
	{
		swap(pNum + max, pNum + idx);
		adjust(pNum, len, max);
	}
	return;
}
void heapSort(int* pNum, int len)
{
	if (pNum == nullptr || len <= 0) return;
	//��ʼ���󶥶�,���һ�����ڵ�Ϊlen/2-1
	for (int i = len / 2 - 1; i >= 0; --i)
	{
		adjust(pNum, len, i);
	}
	for (int i = len - 1; i >= 0; --i)
	{
		swap(pNum+i, pNum);
		//������Ѿ���λ����������ͷ����i��
		adjust(pNum, i, 0);
	}
	return;
}


//	01	string ���� =
class CMyString
{
public:
	CMyString(char* pData = NULL)
	{
		if (!pData)
		{
			m_pData = new char[strlen(pData) + 1];
			memcpy(m_pData, pData, strlen(pData) + 1);
		}
	}
	CMyString(const CMyString& str)
	{
		if (!str.m_pData)
		{
			m_pData = new char[strlen(str.m_pData) + 1];
			memcpy(m_pData, str.m_pData, strlen(str.m_pData) + 1);
		}
	}
	~CMyString()
	{
		if (!m_pData)
		{
			delete[]m_pData;
			m_pData = NULL;
		}
	}
	CMyString& operator=(const CMyString& str)
	{
		if (this!=&str)
		{
			CMyString strTmp(str);
			char* pTmp = m_pData;
			m_pData = strTmp.m_pData;
			strTmp.m_pData = pTmp;
		}
		return *this;
	}
private:
	char* m_pData;
};

//	02	c#

//	03	p39	�������ظ�������	�ܷ��������ռ䣿��ϣ��	�ܷ�ı�����ֵ��   
bool duplicate(int pData[], int len, int* pRepeatNum)
{
	//˼·�����±������ݲ������򽻻�������ͬ�±��е�ֵ
	
	//��Ч��������������ж�
	if (NULL == pData || len <= 1) return false;
	for (int i = 0; i < len; ++i)
	{
		if (pData[i] < 0 || pData[i] >= len) return false;
	}

	//����ʹ��for������while��while���ܳ��ֲ��ɿص����
	for(int cnt = 0;cnt<len;++cnt)
	{
		if (pData[cnt] == pData[pData[cnt]])
		{
			*pRepeatNum = pData[cnt];
			return true;
		}
		while (cnt != pData[cnt])
		{
			if (pData[cnt] == pData[pData[cnt]])
			{
				*pRepeatNum = pData[cnt];
				return true;
			}
			int tmp = pData[pData[cnt]];
			pData[pData[cnt]] = pData[cnt];
			pData[cnt] = tmp;
		}
	}
	return false;
}

//	04	p44	��ά�����еĲ���	ע������ָ��ǿ������ת�������򲽳���һ��
bool find(int* matrix, int rows, int columns, int target)
{
	if (NULL == matrix || 0 >= rows || 0 >= columns || target > *(matrix + rows*columns-1))
		return false;
	//�����Ͻǿ�ʼ
	int row = 0;
	int column = columns - 1;
	while (row < rows && column >= 0)
	{
		if (*(matrix + row*columns+column) == target) return true;
		else if (*(matrix + row*columns + column) > target) --column;
		else ++row;
	}
	return false;
}

//	05	p51	�ո��滻	�ܷ��������ռ�
void replaceBlank(char* str, int reserve,const char* tagStr)
{
	if (NULL == str && reserve <= 0) return;
	int blankNum = 0;
	for (int i = 0; i < strlen(str) + 1; ++i)
	{
		if (str[i] == ' ') ++blankNum;
	}
	//lenΪ��ʹ�õĿռ�
	int tagLen = strlen(tagStr);
	int oldLen = strlen(str);
	int newLen = oldLen + blankNum*(tagLen - 1);
	if (newLen > reserve) return;
	while (newLen >= 0 && newLen > oldLen)
	{
		if (str[oldLen] == ' ')
		{
			for (int i = tagLen - 1; i >= 0; --i)
			{
				str[newLen--] = tagStr[i];
			}
		}
		else
		{
			str[newLen--] = str[oldLen];
		}
		--oldLen;
	}
	return;
}

//	06	p58	��β��ͷ��ӡ����	�ܷ��������ռ�,ѹ��ջ��ȥ���ݹ�ı��ʾ���ջ�ṹ  �ݹ�̫��ջ��� ������ʹ��
void printListReversely(ListNode* head)
{
	if (head == NULL) return;
	printListReversely(head->m_pNext);
	cout << head->m_value << endl;
	return;
}

//	07	p62 �ؽ������� ��֪ ǰ��12473568 ����47215386
BinaryTreeNode* construct(int* const startPreorder, int* const endPreorder, int* const startInorder, int* const endInorder)
{
	if (startPreorder == NULL || endPreorder == NULL || startInorder == NULL || endInorder == NULL) return NULL;

	BinaryTreeNode* root = new BinaryTreeNode(*startPreorder);
	if (startPreorder == endPreorder)
	{
		if (startInorder == endInorder && *startPreorder == *startInorder) return root;
		else return NULL;
	}

	//������������Ҹ��ڵ��ֵ
	int* pRootInoreder = NULL;
	for (int* p = startInorder; p <= endInorder; ++p)
	{
		if (*startPreorder == *p) pRootInoreder = p;
	}
	//����������
	int leftLen = pRootInoreder - startInorder;
	//�����������������β��
	int* pPreorederEnd = startPreorder + leftLen;
	//���ݶ������ı���������ҵĹ����Ȱ���ڵ㶼�����
	if (leftLen > 0) root->m_pLeft = construct(startPreorder + 1, pPreorederEnd, startInorder, pRootInoreder - 1);
	//��ڵ��������ҽڵ�
	if(leftLen<endPreorder-startPreorder) root->m_pRight = construct(pPreorederEnd + 1, endPreorder, pRootInoreder+1, endInorder);
	return root;
}

//	08	p65	����������һ���ڵ�
BinaryTreeNode* getNext(BinaryTreeNode* const pPreNode)
{
	if (NULL == pPreNode) return NULL;

	BinaryTreeNode* pNext = NULL;

	if (NULL != pPreNode->m_pRight)
	{
		pNext = pPreNode->m_pRight->m_pLeft;
		while (pNext->m_pLeft != NULL)
		{
			pNext = pNext->m_pLeft;
		}
		return pNext;
	}
	else if(NULL != pPreNode->m_pParent)
	{
		BinaryTreeNode* pCur = pPreNode;
		pNext = pPreNode->m_pParent;
		while (pNext->m_pLeft != pCur && NULL != pNext->m_pParent)
		{
			pCur = pNext;
			pNext = pNext->m_pParent;
		}
		if (NULL == pNext->m_pParent) return NULL;
		return pNext;
	}
	return pNext;
}

//	09	p68	����ջʵ�ֶ���
template <typename T>
class CQueue
{
public:
	void appendTail(const T& node);
	T popHead();
	T getTail();
private:
	stack<T> m_stack1;
	stack<T> m_stack2;
};

template <typename T>
void CQueue<T>::appendTail(const T& node)
{
	if ( m_stack1.empty() && !m_stack2.empty())
	{
		while (!m_stack2.empty())
		{
			T& data = m_stack2.top();
			m_stack2.pop();
			m_stack1.push(data);
		}
	}
	m_stack1.push(node);
	return;
}

template <typename T>
T CQueue<T>::popHead()
{
	T head = 0;
	if (m_stack2.empty())
	{
		while (!m_stack1.empty())
		{
			T& data = m_stack1.top();
			m_stack1.pop();
			m_stack2.push(data);
		}
	}
	if (m_stack2.empty()) return head;
	head = m_stack2.top();
	m_stack2.pop();
	return head;
}

//	10	p74	쳲���������
long long filbonacci(unsigned int n)
{
	if (0 == n) return 0;
	if (1 == n) return 1;
	long long pre1 = 0;
	long long pre2 = 1;
	long long res = 0;
	for (int i = 0; i < n-1; ++i)
	{
		res = pre1 + pre2;
		pre1 = pre2;
		pre2 = res;
	}
	return res;
}

//	11	p82	��ת�����е���С����
int findMinInOrder(const int* numbers, int length)
{
	if (NULL == numbers || length <= 0) throw "invalid input!";
	for (int i = 1; i < length; ++i)
	{
		if (numbers[i] < numbers[i - 1]) return numbers[i];
	}
	return numbers[0];
}
int findMin(const int* numbers, int length)
{
	if (NULL == numbers || length <=0) throw "invalid input!";
	int index1 = 0;
	int index2 = length-1;
	int indexMid = index1;
	while(numbers[index1] >= numbers[index2])
	{
		if (index2 - index1 == 1)
		{
			indexMid = index2;
			break;
		}
		indexMid = (index1 + index2) / 2;

		if (numbers[index1] == numbers[index2] && numbers[index2] == numbers[indexMid])
		{
			return findMinInOrder(numbers, length);
		}
		if(numbers[indexMid] >= numbers[index1]) index1 = indexMid;
		if(numbers[indexMid] <= numbers[index2]) index2 = indexMid;
	}
	return numbers[indexMid];



	return 0;

}

//	12	p89	�����е�·�� ���ݷ�
bool findAround(const char* const matrix, int rows, int cols, int curRow, int curCol, const char* const path,bool* visited,int& pathPos)
{
	bool res = false;
	if (path[pathPos] == '\0')  return true;
	if (curRow >= 0 && curCol >= 0 && curRow < rows && curCol < cols
		&& (matrix[curRow*cols + curCol] == path[pathPos])
		&& !visited[curRow*cols + curCol])
	{
		++pathPos;
		visited[curRow*cols + curCol] = true;

		res = findAround(matrix, rows, cols, curRow - 1, curCol, path, visited, pathPos) ||
			findAround(matrix, rows, cols, curRow + 1, curCol, path, visited, pathPos) ||
			findAround(matrix, rows, cols, curRow, curCol - 1, path, visited, pathPos) ||
			findAround(matrix, rows, cols, curRow, curCol + 1, path, visited, pathPos);
		if (!res)
		{
			--pathPos;
			visited[curRow*cols + curCol] = false;
		}
	}

	return res;
}
bool containPath(const char* const matrix, int rows, int cols, const char* const path)
{
	if (nullptr == matrix || rows < 1 || cols < 1 || nullptr == path)
	{
		throw "invalid input";
		return false;
	}
	bool* visited = new bool[rows*cols];
	memset(visited, 0, rows*cols);
	int pathPos = 0;
	for (int r = 0; r < rows; ++r)
	{
		for (int c = 0; c < cols; ++c)
		{
			if (findAround(matrix, rows, cols, r, c, path, visited, pathPos))
			{
				delete[]visited;
				return true;
			}
		}
	}

	delete[]visited;
	return false;
}

//	13	p92	�����˵��˶���Χ
unsigned int getBitSum(unsigned int num)
{
	int sum = 0;
	while (num > 0)
	{
		sum += (num % 10);
		num /= 10;
	}
	return sum;
}
bool check(int threshold,int rows,int cols,int r,int c,bool* visited)
{
	if (r >= 0 && r < rows && c >= 0 && c < cols
		&& (getBitSum(r) + getBitSum(c) <= threshold)
		&& !visited[r*cols + c])
	{
		return true;
	}
	return false;
}
int move(int threshold, int rows, int cols, int r, int c, bool* visited)
{
	int step = 0;
	if (r >= 0 && r < rows && c >= 0 && c < cols
		&& (getBitSum(r) + getBitSum(c) <= threshold)
		&& !visited[r*cols + c])
	{
		visited[r*cols + c] = true;
		step = 1 + move(threshold,rows,cols,r-1,c,visited) +
			move(threshold, rows, cols, r, c-1, visited) +
			move(threshold, rows, cols, r+1 - 1, c, visited) +
			move(threshold, rows, cols, r, c+1, visited);
	}
	return step;
}
int reachNum(int threshold, int rows, int cols)
{
	int res = 0;
	if (rows <= 0 && cols <= 0 && threshold < 0) return 0;
	bool* visited = new bool[rows*cols];
	memset(visited, 0, rows*cols);
	res = move(threshold, rows, cols,0,0,visited);
	delete[]visited;
	return res;
}

//	14	p96	������	��̬�滮   f(n) = f(i)*f(n-i)
int maxProduct(int n)
{
	if (n < 2) return 0;
	if (2 == n) return 1;
	if (3 == n) return 2;
	if (n <= 0) throw "invalid input!";
	int max = 0;

	int* products = new int[n+1];
	products[0] = 0;
	products[1] = 1;
	products[2] = 2;
	products[3] = 3;

	for (int i = 4; i <= n; ++i)
	{
		max = 0;
		for (int j = 1; j <= i / 2; ++j)
		{
			int product = products[j] * products[i - j];
			if (product > max) max = product;
			products[i] = max;
		}
	}
	return max;
}

//	15	p100	��������1�ĸ���
int getNumofOne(int decimal)
{
	if (0 == decimal) return 0;
	int num = 0;
	/*
	unsigned int flag = 1;// 00000001 ,��Ҫ�ƶ�ԭ����������һ��ģ��ȥ����ÿһλ�Ƚϣ�����ģ��
	while (flag != 0)
	{
		if (decimal&flag)
		{
			num++;
		}
		flag = flag << 1;
	}
	*/
	while (decimal)
	{
		decimal = decimal&(decimal - 1);
		++num;
	}
	return num;
}

//	16	p110	��ֵ�������η�
bool g_invalidInput = false;
double unsignedPower(double base, int exponent)
{
	if (exponent == 0) return 1.0;
	if (exponent == 1) return base;
	double result = unsignedPower(base, exponent >> 1);
	result *= result;
	if (exponent & 0x01) result *= base;
	return result;
}
double power(double base, int exponent)
{
	if (equal(base, 0.0, 0.001) && exponent < 0)
	{
		g_invalidInput = true;
		return 0.0;
	}
	double uExponent = (exponent < 0) ? (-1)*exponent : exponent;
	
	double result = unsignedPower(base, uExponent);

	if (exponent < 0) return 1.0 / result;
	return result;
}

//	17	p114	��ӡ��1������nλ��
void printNum17(const char* p,int n)
{
	int bit = n-1;
	while (p[bit] == '0')
	{
		--bit;
	}
	while (bit != -1)
	{
		cout << (int)(p[bit--]-'0');
	}
	cout << endl;
	return;
}
void updateNum17(char* p, int n)
{
	int bit = 0;
	while (p[bit] == '9')
	{
		p[bit] = '0';
		bit++;
	}
	p[bit]++;
	return;
}
void printOnetoNbit(int n)
{
	if (n <= 0) return;
	char* pNum = new char[n+1];
	memset(pNum, '0', n+1);
	while (pNum[n] != '1')
	{
		updateNum17(pNum, n);
		printNum17(pNum, n);
	}
	delete[]pNum;
	return;
}

//	18	p119	ɾ������ڵ�
void deleteListNode(ListNode** head, ListNode* pToDelete)
{
	if (nullptr == *head || nullptr == pToDelete) return;

	if (pToDelete->m_pNext != nullptr)
	{
		ListNode* pNext = pToDelete->m_pNext;
		pToDelete->m_value = pNext->m_value;
		pToDelete->m_pNext = pNext->m_pNext;
		delete pNext;
		pNext = nullptr;
	}
	else if (pToDelete == *head)
	{
		delete pToDelete;
		pToDelete = nullptr;
		*head = nullptr;
		return;
	}
	else
	{
		ListNode* pNode = *head;
		while (pNode->m_pNext != pToDelete)
		{
			pNode = pNode->m_pNext;
		}
		pNode->m_pNext = nullptr;
		delete pToDelete;
		pToDelete = nullptr;

	}


	return;
}
	//ɾ�������е��ظ��ڵ�
void deleteDuplication(ListNode** pHead)
{
	
	if (nullptr == *pHead || nullptr == pHead) return;
	ListNode* pPre = nullptr;
	
	ListNode* pNode = *pHead;
	ListNode* pNext = pNode->m_pNext;
	int dupVal = pNode->m_value;
	
	while (dupVal == pNext->m_value)
	{
		delete pNode;
		pNode = pNext;
		pNext = pNode->m_pNext;
		while (dupVal == pNode->m_value)
		{
			delete pNode;
			if (nullptr == pNext)
			{
				*pHead = nullptr;
				return;
			}
			pNode = pNext;
			pNext = pNode->m_pNext;
		}
		dupVal = pNode->m_value;
		if (pNext == nullptr)
		{
			*pHead = pNode;
			return;
		}
	}
	*pHead = pNode;

	pPre = pNode;
	pNode = pNode->m_pNext;
	while (pNode->m_pNext != nullptr)
	{
		pNext = pNode->m_pNext;
		if (pNode->m_value != pNext->m_value)
		{
			pPre = pNode;
			pNode = pNode->m_pNext;
			pNext = pNode->m_pNext;
			continue;
		}
		while (pNext!= nullptr && pNode->m_value == pNext->m_value)
		{
			pNode->m_pNext = pNext->m_pNext;
			delete pNext;
			pNext = nullptr;
			pNext = pNode->m_pNext;
		}
		delete pPre->m_pNext;
		pPre->m_pNext = pNext;
		if (pNext == nullptr || pNext->m_pNext == nullptr) return;
		pNode = pNext;
		pNext = pNode->m_pNext;
	}
	return;
}

//	19	p124	������ʽƥ��
bool matchRecur(char* str, char* pattern)
{
	if (*str == '\0' && *pattern == '\0') return true;
	if (*str != '\0' && *pattern == '\0') return false;
	if (*(pattern+1) == '*')
	{
		if (*str == *pattern || (*(pattern) == '.'&&str!='\0'))
		{
			return matchRecur(str + 1, pattern + 2) || \
				matchRecur(str, pattern + 2) || \
				matchRecur(str + 1, pattern);
		}
		else
		{
			return matchRecur(str, pattern + 2);
		}
	}
	if ((*str == *pattern || (*(pattern) == '.'&&str != '\0')))
	{
		return matchRecur(str+1, pattern + 1);
	}
	return false;
}
bool match(char* str, char* pattern)
{
	if (str == nullptr || pattern == nullptr) return false;
	int strLen = strlen(str);
	int cntStar = 0;
	for (int i = 0; i < strlen(pattern); ++i)
	{
		if (pattern[i] == '*') ++cntStar;
	}
	int patternLen = strlen(pattern);
	if (strLen < patternLen - 2 * cntStar) return false;

	return matchRecur(str,pattern);
}

//	20	p127	��ʾ��ֵ���ַ���
bool isNumeric(const char* str)
{
	if (nullptr == str) return false;

	//�ж���������
	if (*str == '+' || *str == '-') ++str;

	while (*str != '.' && *str != '\0')
	{
		if (*str < '0' || *str>'9')	return false;
		++str;
	}
	if (*str == '\0') return true;

	//�ж�С������
	if (*str == '.')
	{
		++str;
		while (*str != 'e' && *str != 'E' && *str != '\0')
		{
			if (*str < '0' || *str>'9')	return false;
			++str;
		}
	}
	if (*str == '\0') return true;
	//�ж�ָ������
	if (*str == 'e' || *str == 'E')
	{
		++str;
		if (*str == '+' || *str == '-') ++str;
		while (*str != '\0')
		{
			if (*str < '0' || *str>'9')	return false;
			++str;
		}
		if (*str == '\0') return true;

	}
	if (*str == '\0') return true;
	
	return false;
}

//	21	p129	��������˳��ʹ������λ��ż��ǰ��
bool isOdd(int num)
{
	return (num & 0x01);
}
void oddBeforeEven(int* pNum,int len,bool(*Func)(int))
{
	if (pNum == nullptr || len <= 0) return;
	int* pHead = pNum;
	int* pTail = pNum + len - 1;
	while (pHead < pTail)
	{
		while (pHead < pTail && Func(*pHead)) ++pHead;
		while (pHead < pTail && !Func(*pTail)) --pTail;
		if (pHead < pTail)
		{
			int tmp = *pHead;
			*pHead = *pTail;
			*pTail = tmp;
		}
	}
	return;
}

//	22	p134	�����е�����k���ڵ�	�������³����  �Ⱥ�ָ��
void reverseRecur(ListNode* pNode,ListNode** pRes, unsigned int* p, unsigned int k)
{
	if (pNode == nullptr)
	{
		*p = 0;
		return;
	}
	reverseRecur(pNode->m_pNext, pRes, p,k);
	++(*p);
	if (*p == k) *pRes = pNode;
	return;
}
ListNode* findReverseK(ListNode* pHead, unsigned int k)
{
	ListNode* pNode = nullptr;
	unsigned int i = 0;
	reverseRecur(pHead, &pNode, &i,k);
	return pNode;
}
ListNode* findKthToTail(ListNode* pHead, unsigned int k)
{
	if (pHead == nullptr || k == 0) return nullptr;
	ListNode* pAhead = pHead;
	ListNode* pAfter = pHead;

	for (int i = 0; i < k - 1; ++i)//���kΪ0�Ļ���k-1��Ϊ2^32-1
	{
		if (pAhead == nullptr) return nullptr;
		pAhead = pAhead->m_pNext;
	}

	while (pAhead->m_pNext != nullptr)
	{
		pAhead = pAhead->m_pNext;
		pAfter = pAfter->m_pNext;
	}
	return pAfter;

}

//	23	p139	�����л������	����ָ��
int getNumOfLoop(ListNode* pHead)
{
	if (nullptr == pHead) return 0;
	ListNode* pFast = pHead;
	ListNode* pSlow = pHead;
	while (pFast != pSlow)
	{
		if (pFast == nullptr) return 0;
		pFast = pFast->m_pNext;
		if (pFast == nullptr) return 0;
		pFast = pFast->m_pNext;
		pSlow = pSlow->m_pNext;
	}

	int cnt = 1;
	pFast = pFast->m_pNext->m_pNext;
	pSlow = pSlow->m_pNext;
	while (pFast != pSlow)
	{
		pFast = pFast->m_pNext->m_pNext;
		pSlow = pSlow->m_pNext;
		++cnt;
	}
	return cnt;
}
ListNode* findEntrance(ListNode* pHead)
{
	int n = getNumOfLoop(pHead);
	if (0 == n) return nullptr;
	if (nullptr == pHead) return nullptr;
	ListNode* pAhead = pHead;
	ListNode* pAfter = pHead;

	for (int i = 0; i < n; ++i)
	{
		pAhead = pAhead->m_pNext;
	}

	while (pAhead != pAfter)
	{
		pAfter = pAfter->m_pNext;
		pAhead = pAhead->m_pNext;
	}
	return pAhead;
}

//	24	p142	��ת����
ListNode* reverseList(ListNode* pHead)
{
	if (nullptr == pHead) return nullptr;
	ListNode* pPre = nullptr;
	ListNode* pNext = nullptr;
	while (pHead != nullptr)
	{
		pNext = pHead->m_pNext;
		pHead->m_pNext = pPre;
		pPre = pHead;
		pHead = pNext;
	}
	return pPre;
}

//	25	p145	�ϲ��������������
ListNode* mergeTwoOrderedList(ListNode* pA, ListNode* pB)
{
	if (pA == nullptr && pB == nullptr) return nullptr;
	if (pA == nullptr && pB != nullptr) return pB;
	if (pA != nullptr && pB == nullptr) return pA;
	ListNode* pHead = nullptr;
	if (pA->m_value <= pB->m_value)
	{
		pHead = pA;
		pA = pA->m_pNext;
	}
	else
	{
		pHead = pB;
		pB = pB->m_pNext;
	}
	ListNode* pNode = pHead;
	while (pA != nullptr && pB != nullptr)
	{
		if (pA->m_value <= pB->m_value)
		{
			pNode->m_pNext = pA;
			pNode = pA;
			pA = pA->m_pNext;
		}
		else
		{
			pNode->m_pNext = pB;
			pNode = pB;
			pB = pB->m_pNext;
		}
	}
	if (pA == nullptr)
	{
		pNode->m_pNext = pB;
	}
	else
	{
		pNode->m_pNext = pA;
	}
	return pHead;
}

//	26	p148	�����ӽṹ
bool isSameTree(BinaryTreeNode* pA, BinaryTreeNode* pB)
{
	if (pB == nullptr) return true;
	if (pA == nullptr) return false;
	if (pA->m_value  == pB->m_value)
	{
		if (isSameTree(pA->m_pLeft, pB->m_pLeft) && isSameTree(pA->m_pRight, pB->m_pRight))
			return true;
	}
	return false;
}
bool isChildTree(BinaryTreeNode* pA, BinaryTreeNode* pB)
{
	if (pA == nullptr || pB == nullptr) return false;
	bool isChild = false;

	if (pA->m_value == pB->m_value) isChild = isSameTree(pA, pB);

	if(!isChild) isChild = isChildTree(pA->m_pLeft, pB) || isChildTree(pA->m_pRight, pB);

	return isChild;
}

//	27	p157	�������ľ���
void myMirrorTreeRecur(BinaryTreeNode* pRoot)
{
	if (pRoot == nullptr) return;
	if (pRoot->m_pLeft == nullptr && pRoot->m_pRight == nullptr) return;
	if (pRoot->m_pLeft != nullptr && pRoot->m_pRight == nullptr) myMirrorTreeRecur(pRoot->m_pLeft);
	else if (pRoot->m_pLeft == nullptr && pRoot->m_pRight != nullptr) myMirrorTreeRecur(pRoot->m_pRight);
	else
	{
		myMirrorTreeRecur(pRoot->m_pLeft);
		myMirrorTreeRecur(pRoot->m_pRight);
	}
	BinaryTreeNode* pTmp = pRoot->m_pLeft;
	pRoot->m_pLeft = pRoot->m_pRight;
	pRoot->m_pRight = pTmp;
	return;
}
void offerMirrorTreeRecur(BinaryTreeNode* pRoot)
{
	if (pRoot == nullptr) return;
	if (pRoot->m_pLeft == nullptr && pRoot->m_pRight == nullptr) return;
	BinaryTreeNode* pTmp = pRoot->m_pLeft;
	pRoot->m_pLeft = pRoot->m_pRight;
	pRoot->m_pRight = pTmp;
	if (pRoot->m_pLeft) offerMirrorTreeRecur(pRoot->m_pLeft);
	if (pRoot->m_pRight) offerMirrorTreeRecur(pRoot->m_pRight);
}

//	28	p159	�Գƶ�����	һֱ�ھ�����ô������ͬ�����У�����������Ϊ���ڵ������ѽ
bool isMirrorTree(BinaryTreeNode* pRoot1, BinaryTreeNode* pRoot2)
{
	if (pRoot1 == nullptr && pRoot2 == nullptr) return true;
	if (pRoot1 == nullptr || pRoot2 == nullptr) return false;
	if (pRoot1->m_value != pRoot2->m_value) return false;
	return isMirrorTree(pRoot1->m_pLeft, pRoot2->m_pRight) && isMirrorTree(pRoot1->m_pRight, pRoot2->m_pLeft);
}
bool myIsMirrorTree(BinaryTreeNode* pRoot)
{
	if (nullptr == pRoot) return false;
	return isMirrorTree(pRoot, pRoot);

}

//	29	p161	˳ʱ���ӡ����
void myPrintMatrixClockwisely(int* matrix, const int cRows, const int cCols,int rows,int cols)
{
	if (nullptr == matrix || 0 == rows || 0 == cols) return;
	if (rows == 1)
	{
		for (int i = 0; i < cols; ++i) cout << *(matrix + i) << " ";
		cout << endl;
		return;
	}
	if (cols == 1)
	{
		for (int i = 0; i < rows; ++i) cout << *(matrix + i) << " ";
		cout << endl;
		return;
	}
	if(cols == 2 && rows ==2)
	{
		cout << matrix[0] << " " << matrix[1] << " " << matrix[cCols + 1] << " " << matrix[cCols] << endl;
		return;
	}

	for (int i = 0; i < cols; ++i) cout << matrix[i] << " ";
	for (int i = 0; i < rows - 2; ++i) cout << matrix[(i+1)*cCols + cols-1] << " ";
	for (int i = 0; i < cols; ++i) cout << matrix[(rows-1)*cCols + cols-1-i] << " ";
	for (int i = 0; i < rows - 2; ++i) cout << matrix[(rows - 2-i)*cCols] << " ";

	myPrintMatrixClockwisely(matrix + cols + 1, cRows, cCols,rows - 2, cols - 2);
	return;
}

//	30	p165	����min������ջ
template<typename T>
class StackWithMin
{
public:
	void push(const T& value);
	void pop();
	T min();
private:
	stack<T> m_data;
	stack<T> m_min;
};

template<typename T>
void StackWithMin<T>::push(const T& value) 
{
	m_data.push(value);
	if (m_min.empty()) m_min.push(value);
	else
	{
		if (value < m_min.top()) m_min.push(value);
		else m_min.push(m_min.top());
	}
	return;
}

template<typename T>
void StackWithMin<T>::pop()
{
	if (m_data.empty() || m_min.empty()) return;
	m_data.pop();
	m_min.pop();
	return;
}

template<typename T>
T StackWithMin<T>::min()
{
	assert(!(m_data.empty() || m_min.empty()));
	return m_min.top();
}

//	31	p168	ջ��ѹ�뵯������
bool isPushPopMatch(const int* pPush, const int* pPop, int len)
{
	if (pPush == nullptr || pPop == nullptr || len <= 0) return false;
	stack<int> sTmp;
	for (int i = 0; i < len; ++i)
	{
		if (sTmp.empty())
		{
			sTmp.push(pPush[i]);
		}
		else if(sTmp.top() != *pPop)
		{
			sTmp.push(pPush[i]);
		}
		else
		{
			while (sTmp.top() == *pPop)
			{
				sTmp.pop();
				++pPop;
			}
			--i;
		}
	}
	while (!sTmp.empty())
	{
		if (sTmp.top() == *pPop)
		{
			sTmp.pop();
			++pPop;
		}
		else break;
	}
	return	sTmp.empty();
}

//	32	p171	����ӡ������
void printByLayer(BinaryTreeNode* pRoot)
{
	if (nullptr == pRoot) return;
	queue<BinaryTreeNode*> qLayer;
	qLayer.push(pRoot);
	vector<int> vecNode;
	vecNode.push_back(pRoot->m_value);

	while (!qLayer.empty())
	{
		for (int i = 0; i < qLayer.size(); ++i)
		{
			BinaryTreeNode* pTmp = qLayer.front();
			if (pTmp->m_pLeft)
			{
				qLayer.push(pTmp->m_pLeft);
				vecNode.push_back(pTmp->m_pLeft->m_value);
			}
			if (pTmp->m_pRight)
			{
				qLayer.push(pTmp->m_pRight);
				vecNode.push_back(pTmp->m_pRight->m_value);
			}
			qLayer.pop();
		}
	}

	for (vector<int>::iterator it = vecNode.begin(); it != vecNode.end(); ++it)
	{
		cout << *it << " ";
	}
	cout << endl;
	return;
}

//	33	p179	�������������������
bool isBST(int* num, int len)
{
	if (nullptr == num || len <= 0) return false;

	int root = num[len - 1];

	int i = 0;
	for (; i < len - 1; ++i)
	{
		if (num[i] > root) break;
	}
	int j = i;
	for (; j < len - 1; ++j)
	{
		if (num[j] < root) return false;
	}

	bool left = false;
	if (i > 0) left = isBST(num, i);

	bool right = false;
	if (i < len - 1) right = isBST(num + i, len - i - 1);

	return (left&&right);
}

//	34	p182	�������к�Ϊĳһ��ֵ��·��
void getPathRecur(BinaryTreeNode* pRoot,vector<int>& vPath,int sumTmp,const int sum)
{
	if (pRoot == nullptr) return;
	sumTmp += pRoot->m_value;
	vPath.push_back(pRoot->m_value);
	getPathRecur(pRoot->m_pLeft, vPath, sumTmp, sum);
	getPathRecur(pRoot->m_pRight, vPath, sumTmp, sum);
	if (sumTmp == sum)	printVector<int>(vPath);
	else
	{
		sumTmp -= pRoot->m_value;
		vPath.pop_back();
	}
	return;
}
void getPathOfSum(BinaryTreeNode* pRoot, const int sum)
{
	if (nullptr == pRoot) return;

	vector<int> vPath;

	int sumTmp = 0;

	getPathRecur(pRoot, vPath, sumTmp, sum);

	return;
}

//	35	p187	��������ĸ���
ComplexListNode* complexListClone(ComplexListNode* pHead)
{
	if (pHead == nullptr) return nullptr;
	ComplexListNode* pTmpHead = pHead;
	ComplexListNode* pAns = new ComplexListNode(pTmpHead->m_value);
	ComplexListNode* pTmpAns = pAns;
	ComplexListNode* pNode;
	unordered_map<int, ComplexListNode*> mComplexList;
	unordered_map<int, ComplexListNode*>::iterator it;
	
	while (pTmpHead->m_pNext != nullptr)
	{
	
		it = mComplexList.find(pTmpHead->m_pNext->m_value);
		if (it == mComplexList.end())
		{
			pNode = new ComplexListNode(pTmpHead->m_pNext->m_value);
			mComplexList.insert(pair<int, ComplexListNode*>(pTmpHead->m_pNext->m_value,pNode));
			pTmpAns->m_pNext = pNode;
		}
		else
		{
			pTmpAns->m_pNext = mComplexList[pTmpHead->m_pNext->m_value];
		}

		if (pTmpHead->m_pSibling != nullptr)
		{
			it = mComplexList.find(pTmpHead->m_pSibling->m_value);
			if (it == mComplexList.end())
			{
				pNode = new ComplexListNode(pTmpHead->m_pSibling->m_value);
				mComplexList.insert(pair<int, ComplexListNode*>(pTmpHead->m_pSibling->m_value, pNode));
				pTmpAns->m_pSibling = pNode;
			}
			else
			{
				pTmpAns->m_pSibling = mComplexList[pTmpHead->m_pSibling->m_value];
			}
		}
		
		pTmpAns = pTmpAns->m_pNext;
		pTmpHead = pTmpHead->m_pNext;
	}
	if (pTmpHead->m_pSibling != nullptr)
	{
		it = mComplexList.find(pTmpHead->m_pSibling->m_value);
		if (it == mComplexList.end())
		{
			pNode = new ComplexListNode(pTmpHead->m_pSibling->m_value);
			mComplexList.insert(pair<int, ComplexListNode*>(pTmpHead->m_pSibling->m_value, pNode));
			pTmpAns->m_pSibling = pNode;
		}
		else
		{
			pTmpAns->m_pSibling = mComplexList[pTmpHead->m_pSibling->m_value];
		}
	}
	return pAns;
}

//	36	p191	������������˫������
void convertRecur(BinaryTreeNode* pNode, BinaryTreeNode** pLast)
{
	if (nullptr == pNode) return;

	if (pNode->m_pLeft != nullptr)
		convertRecur(pNode->m_pLeft, pLast);

	pNode->m_pLeft = *pLast;

	if (*pLast != nullptr)
		(*pLast)->m_pRight = pNode;

	*pLast = pNode;

	if (pNode->m_pRight != nullptr)
		convertRecur(pNode->m_pRight, pLast);

	return;
}
BinaryTreeNode* BSTtoList(BinaryTreeNode* pRoot)
{
	BinaryTreeNode* pLastNode = nullptr;

	convertRecur(pRoot, &pLastNode);

	BinaryTreeNode* pListHead = pLastNode;

	while (pListHead != nullptr && pListHead->m_pLeft != nullptr)
	{
		pListHead = pListHead->m_pLeft;
	}
	return pListHead;

}

//	37	p194	���л�������

//	38	p197	�ַ�����ȫ����
void permutation(char* pStr, char* pBegin)
{
	if (*pBegin == '\0')
	{
		string str(pStr);
		cout << str << endl;
	}
	else
	{
		for (char* pCh = pBegin; *pCh != '\0'; ++pCh)
		{
			char tmp = *pCh;
			*pCh = *pBegin;
			*pBegin = tmp;

			permutation(pStr, pBegin + 1);

			tmp = *pCh;
			*pCh = *pBegin;
			*pBegin = tmp;

		}
	}

}
void permutation(char* pStr)
{
	if (pStr == nullptr) return;

	permutation(pStr, pStr);
}

	//	stl ʵ��
void permutationSTL(char* pStr)
{
	do
	{
		string str(pStr);
		cout << str << endl;
	} while (next_permutation(pStr, pStr + strlen(pStr)));//Ҫ������������������
	return;
}

	//	8 queen
bool eightQueenJudge(int* pNum,int len)
{
	for (int i = 0; i < len; ++i)
	{
		for (int j = i + 1; j < len; ++j)
		{
			if (abs(i - j) == abs(pNum[i] - pNum[j])) return false;
		}
	}
	return true;
}

int eightQueen(int* pNum,int len)
{
	int cnt = 0;
	do
	{
		if (eightQueenJudge(pNum, len)) ++cnt;
	}while(next_permutation(pNum, pNum + len));
	return cnt;
}

//	39	p205	�����г��ִ�������һ�������
int findNumOverHalf(int* pNum, int len)
{
	assert(pNum != nullptr && len > 0);
	unordered_map<int, int> umNum;
	for (int i = 0; i < len; ++i)
	{
		if (umNum.find(pNum[i]) == umNum.end())
		{
			umNum.insert(pair<int, int>(pNum[i], 1));
		}
		else
		{
			umNum[pNum[i]]++;
		}
	}
	unordered_map<int, int>::iterator it = umNum.begin();
	for (it = umNum.begin(); it != umNum.end(); ++it)
	{
		//cout << (*it).first << " " << (*it).second << endl;
		if ((*it).second > (len / 2)) return (*it).first;
	}

	return -1;
}
