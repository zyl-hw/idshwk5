from sklearn.ensemble import RandomForestClassifier
import numpy as np

def entropy(s):
    _, counts = np.unique(list(s), return_counts=True)
    total = sum(counts)
    percent = list(map(lambda x: x / total, counts))
    return sum(-n * np.log(n) for n in percent)
    
def processDomain(s):
    return len(s), sum(c.isdigit() for c in s), entropy(s)

class Domain:
    def __init__(self, _domain, _label):
        self.domain = _domain
        self.label = _label
        self.domainNameLength, self.domainNumberCount, self.letterEntropy = processDomain(self.domain)

    def returnData(self):
        return [self.domainNameLength, self.domainNumberCount, self.letterEntropy]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

domainList = []      

def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tokens = line.split(",")
            domain = tokens[0]
            label = tokens[1]
            domainList.append(Domain(domain, label))

def initTest(filename):
    with open(filename) as f:
        return list(filter(lambda line: line != "", map(lambda line: line.strip(), f)))

if __name__ == '__main__':
    initData("train.txt")
    featureMatrix = list(map(lambda domain: domain.returnData(), domainList))
    labelList = list(map(lambda domain: domain.returnLabel(), domainList))

    testDomains = initTest("test.txt")
    testDomainFeatures = []
    for domain in testDomains:
        domainNameLength, domainNumberCount, letterEntropy = processDomain(domain)
        testDomainFeatures.append([domainNameLength, domainNumberCount, letterEntropy])

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    testLabels = clf.predict(testDomainFeatures)
    output = list(zip(testDomains, testLabels))
    with open("result.txt", "w+") as f:
        for domain, label in output:
            line = domain + ","
            if label == 0:
                line = line + "notdga\n"
            else:
                line = line + "dga\n"
            f.write(line)
