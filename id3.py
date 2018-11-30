import re
import math
from collections import deque

class Node(object):
	def __init__(self):
		self.value = None
		self.next = None
		self.childs = None

class DecisionTree(object):
	def __init__(self, sample, attributes, labels):
		self.sample = sample
		self.attributes = attributes
		self.labels = labels
		self.labelCodes = None
		self.labelCodesCount = None
		self.initLabelCodes()
		self.root = None
		self.entropy = self.getEntropy([x for x in range(len(self.labels))])

	def initLabelCodes(self):
		self.labelCodes = []
		self.labelCodesCount = []
		for l in self.labels:
			if l not in self.labelCodes:
				self.labelCodes.append(l)
				self.labelCodesCount.append(0)
			self.labelCodesCount[self.labelCodes.index(l)] += 1

	def getLabelCodeId(self, sampleId):
		return self.labelCodes.index(self.labels[sampleId])

	def getAttributeValues(self, sampleIds, attributeId):
		vals = []
		for sid in sampleIds:
			val = self.sample[sid][attributeId]
			if val not in vals:
				vals.append(val)
		return vals

	def getEntropy(self, sampleIds):
		entropy = 0
		labelCount = [0] * len(self.labelCodes)
		for sid in sampleIds:
			labelCount[self.getLabelCodeId(sid)] += 1
		for lv in labelCount:
			if lv != 0:
				entropy += -lv/len(sampleIds) * math.log(lv/len(sampleIds), 2)
			else:
				entropy += 0
		return entropy

	def getDominantLabel(self, sampleIds):
		labelCodesCount = [0] * len(self.labelCodes)
		for sid in sampleIds:
			labelCodesCount[self.labelCodes.index(self.labels[sid])] += 1
		return self.labelCodes[labelCodesCount.index(max(labelCodesCount))]

	def getInformationGain(self, sampleIds, attributeId):
		gain = self.getEntropy(sampleIds)
		attributeVals = []
		attributeValsCount = []
		attributeValsIds = []
		for sid in sampleIds:
			val = self.sample[sid][attributeId]
			if val not in attributeVals:
				attributeVals.append(val)
				attributeValsCount.append(0)
				attributeValsIds.append([])
			vid = attributeVals.index(val)
			attributeValsCount[vid] += 1
			attributeValsIds[vid].append(sid)
		for vc, vids in zip(attributeValsCount, attributeValsIds):
			gain -= vc/len(sampleIds) * self.getEntropy(vids)
		return gain

	def getAttributeMaxInformationGain(self, sampleIds, attributeIds):
		attributesEntropy = [0] * len(attributeIds)
		for i, attId in zip(range(len(attributeIds)), attributeIds):
			attributesEntropy[i] = self.getInformationGain(sampleIds, attId)
		maxId = attributeIds[attributesEntropy.index(max(attributesEntropy))]
		return self.attributes[maxId], maxId

	def isSingleLabeled(self, sampleIds):
		label = self.labels[sampleIds[0]]
		for sid in sampleIds:
			if self.labels[sid] != label:
				return False
		return True

	def getLabel(self, sampleId):
		return self.labels[sampleId]

	def id3(self):
		sampleIds = [x for x in range(len(self.sample))]
		attributeIds = [x for x in range(len(self.attributes))]
		self.root = self.id3Recv(sampleIds, attributeIds, self.root)

	def id3Recv(self, sampleIds, attributeIds, root):
		root = Node()
		if self.isSingleLabeled(sampleIds):
			root.value = self.labels[sampleIds[0]]
			return root
		if len(attributeIds) == 0:
			root.value = self.getDominantLabel(sampleIds)
			return root
		bestAttrName, bestAttrId = self.getAttributeMaxInformationGain(
			sampleIds, attributeIds)
		root.value = bestAttrName
		root.childs = []
		for value in self.getAttributeValues(sampleIds, bestAttrId):
			child = Node()
			child.value = value
			root.childs.append(child)
			childSampleIds = []
			for sid in sampleIds:
				if self.sample[sid][bestAttrId] == value:
					childSampleIds.append(sid)
			if len(childSampleIds) == 0:
				child.next = self.getDominantLabel(sampleIds)
			else:
				if len(attributeIds) > 0 and bestAttrId in attributeIds:
					toRemove = attributeIds.index(bestAttrId)
					attributeIds.pop(toRemove)
				child.next = self.id3Recv(
					childSampleIds, attributeIds, child.next)
		return root

	def printTree(self):
		if self.root:
			roots = deque()
			roots.append(self.root)
			while len(roots) > 0:
				root = roots.popleft()
				print(root.value)
				if root.childs:
					for child in root.childs:
						print('({})'.format(child.value))
						roots.append(child.next)
				elif root.next:
					print(root.next)


def main():
	f = open('playtennis.csv')
	attributes = f.readline().split(',')
	attributes = attributes[1:len(attributes)-1]
	print(attributes)
	sample = f.readlines()
	f.close()
	for i in range(len(sample)):
		sample[i] = re.sub('\d+,', '', sample[i])
		sample[i] = sample[i].strip().split(',')
	labels = []
	for s in sample:
		labels.append(s.pop())
	decisionTree = DecisionTree(sample, attributes, labels)
	decisionTree.id3()
	decisionTree.printTree()


main()