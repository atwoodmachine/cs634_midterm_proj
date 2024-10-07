import time
import pandas as pd
from itertools import combinations
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

#Represents a k-itemset
class ItemSet:
    def __init__(self, items):
        self.items = items #a set of strings (item names)
        self.support = 0
        self.supportPercent = 0
        self.allAntecedents = [] #a list of sets of item names for association rules
    def __str__(self):
        return f"{self.items} Support: {self.support}, {self.supportPercent}%"

#Represents an association rule
class AssociationRule:
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent #left hand side of rule
        self.consequent = consequent #right hand side of rule
        self.confidence = 0 #a percentage
        self.ruleSupport = 0 #a percentage
    def __str__(self):
        return f"{self.antecedent} -> {self.consequent} Support: {self.ruleSupport}%, Confidence: {self.confidence}%"

#Helper function to calculates support for itemsets
#Takes a set of ItemSet objects
def itemset_support(itemSet):
    #process given itemSet and remove sets with less than minimum support
    for items in itemSet:
        for trans in transactionsList:
            compare = items.items
            if(compare.issubset(trans)):
                items.support += 1
    for item in itemSet:
        item.supportPercent = item.support/total_transactions * 100

#Helper function to remove infrequent itemsets
def frequent_itemset(itemSet):
    ret = []
    for item in itemSet:
        if(item.supportPercent >= minSupport):
            ret.append(item)
    return ret

#Helper function to generate all antecedents possible for an item set
def antecedents(itemSet):
    temp = []
    ret = []
    for string in itemSet.items:
        temp.append(string)
    for i in range(1, len(temp)):
        for combo in combinations(temp, i):
            ret.append(set(combo))
    itemSet.allAntecedents = ret

#Returns frequent k-itemset by making every combination from the 1 itemset.
#Returns a list of ItemSet objects 
def k_itemset(k, singleItems):
    temp = []
    for thing in singleItems:
        for name in thing.items:
            temp.append(name)
    combos = list(combinations(temp, k))  
    ret = []
    for combo in combos:
        tempItem = ItemSet(set(combo))
        ret.append(tempItem)
    itemset_support(ret)
    frret = frequent_itemset(ret)
    return frret

#Helper function, takes a set of strings (item names)
def findSupport(find):
    for item in allGeneratedItemsets:
        if(item.items == find):
            return item.support
    return 0 

#Generates association rules and calculates confidence
#Takes an ItemSet object
def association_Rules(generateFrom):
    ret = [] #list of association rule objects
    numerator = generateFrom.support
    denominator = 0
    antecedents(generateFrom)
    ants = generateFrom.allAntecedents
    for ant in ants:
        consequent = generateFrom.items.difference(ant)
        temp = AssociationRule(ant, consequent)
        denominator = findSupport(ant)
        temp.ruleSupport = generateFrom.supportPercent
        temp.confidence = numerator/denominator * 100
        ret.append(temp)
    return ret
    
#Intro
print("Available stores are listed below.\n")
print("1. Barnes and Noble\n2. Citadel Paints\n3. GameStop\n4. Staples\n5. Warhammer\n")

itemsAvailable = ''
transactions = ''

#Prompt user for valid selection
while(True):
    selection = input("Enter a number to select one of these stores, or enter 'q' to quit:\n")
    if(selection == 'q'):
        exit()

    selection = int(selection)

    if(selection == 1):
        itemsAvailable = 'barnes_and_noble.csv'
        transactions = 'barnes_and_noble_transactions.csv'
        break
    elif(selection == 2):
        itemsAvailable = 'citadel_paints.csv'
        transactions = 'citadel_paints_transactions.csv'
        break
    elif(selection == 3):
        itemsAvailable = 'gamestop.csv'
        transactions = 'gamestop_transactions.csv'
        break
    elif(selection == 4):
        itemsAvailable = 'staples.csv'
        transactions = 'staples_transactions.csv'
        break
    elif(selection == 5):
        itemsAvailable = 'warhammer.csv'
        transactions = 'warhammer_transactions.csv'
        break
    else:
        print("Invalid selection")

#Prompt user for support and confidence
while(True):
    minSupport = float(input("Please enter a minimum support value from 1 to 100 (this value is interpreted as a percentage):\n"))
    if(minSupport < 1 or minSupport > 100):
        print("Please enter a valid input from 1 to 100")
        continue
    break

while(True):
    minConfidence = float(input("Please enter a minimum confidence value from 1 to 100 (this value is interpreted as a percentage):\n"))
    if(minConfidence < 1 or minConfidence > 100):
        print("Please enter a valid input from 1 to 100")
        continue
    break

#User defined input files
read_items = pd.read_csv(itemsAvailable, usecols=[1]) #Only item names are relevant, expect item names to be unique 
read_transactions = pd.read_csv(transactions)

total_transactions = len(read_transactions)

bruteStartTime = time.time()

#preprocess transactions into a list of sets (for usage of subset methods)
transactionsList = []
for x in range(total_transactions):
    tempTransaction = read_transactions.at[x, 'Transaction']
    tempTransaction = tempTransaction.split(', ')
    transactionsList.append(set(tempTransaction))

#Initial 1-itemsets in a list
oneItemsets = []
for i in range(len(read_items)):
    itemsInSet = {read_items.at[i, 'Item Name']}
    tempSet = ItemSet(itemsInSet)
    oneItemsets.append(tempSet)

itemset_support(oneItemsets)
frequent_itemset(oneItemsets)

allGeneratedItemsets = []
tempSets = []

#Generate frequent k-itemsets
for i in range(1, len(read_items)):
    tempSets = k_itemset(i, oneItemsets)
    if(len(tempSets) == 0):
        break 
    else:
        allGeneratedItemsets += tempSets

#Generate association rules
assocRules = []
for item in allGeneratedItemsets:
    assocRules += association_Rules(item)

if(len(assocRules) == 0):
    print("No rules found. Try a lower minimum support value (25 or less is realistic for these data sets) or a lower minimum confidence level (60 or less is realistic for these data sets)")
else:
    print("Association rules (X -> Y read 'X implies Y'):")

for rule in assocRules:
    if(rule.confidence >= minConfidence):
        print(rule)
bruteEndTime = time.time()

print('\nResults returned in ', (bruteEndTime - bruteStartTime),'seconds from brute force algorithm\n')

#apriori from mlxtend for comparison
#data preprocessing for mlxtend
print('mlxtend Implementations for comparison purposes:\n')
aprioriStart = time.time()
dataset = []
for x in range(total_transactions):
    tempTrans= read_transactions.at[x, 'Transaction']
    tempTrans = tempTrans.split(', ')
    dataset.append(list(tempTrans))

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

apriori_generated = apriori(df, min_support=(minSupport/100), use_colnames=True)

rules = association_rules(apriori_generated, metric="confidence", min_threshold=(minConfidence/100))
rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
aprioriEnd = time.time()
print("Association rules: ", rules, "\n")
print('Results returned in ', (aprioriEnd - aprioriStart),'seconds from mlxtend apriori implementation\n')


#fpgrowth from mlxtend for comparison

fpStart = time.time()
fpgrowth_generated = fpgrowth(df, min_support=(minSupport/100), use_colnames=True)
rules = association_rules(apriori_generated, metric="confidence", min_threshold=(minConfidence/100))
rules = rules[['antecedents', 'consequents', 'support', 'confidence']]
fpEnd = time.time()
print("Association rules: ", rules, "\n")
print('Results returned in ', (fpEnd - fpStart),'seconds from mlxtend fpgrowth implementation')