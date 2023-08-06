import itertools
import random
import string
import sys
from copy import deepcopy
from itertools import combinations


class Term:
    def __init__(self, name, type, arguments=[]):
        self.name = name
        self.type = type
        self.arguments = arguments

    def __str__(self):
        return self.name+str(tuple(self.arguments)).replace("'", "")

class Predicate:
    def __init__(self, name, negation, arguments=[]):
        self.name = name
        self.negation = negation
        self.arguments = arguments

    def __str__(self):
        finalStr = self.name+str(tuple(str(a) for a in self.arguments))
        finalStr = '!' + finalStr if self.negation else finalStr.replace("'", "")
        finalStr = finalStr.replace("(),", "")
        finalStr = finalStr.replace("()", "")
        return finalStr


class Clause:
    def __init__(self, predicates = []):
        self.resolver = set()
        self.predicates = predicates
        self.seen = []

    def __str__(self):
        clause = ""
        for p in self.predicates:
            clause += str(p) + " "
        return clause.strip()

    def addPredicate(self, predicate):
        self.predicates.append(predicate)
        self.predicates.sort(key=lambda x:x.name)

    def setPredicates(self, predicates: list):
        self.predicates = predicates
        self.predicates.sort(key=lambda x: x.name)

    def equal(self, other):
        predsInFirst = sorted(self.predicates, key=lambda p: p.name)
        predsInSecond = sorted(other.predicates, key=lambda p: p.name)

        for i in range(max(len(predsInFirst), len(predsInSecond))):
            if not (predsInFirst[i].name == predsInSecond[i].name and
                    predsInFirst[i].negation == predsInSecond[i].negation):
                return False
        return True


def getValuesFromFile(fileName):
    constants = []
    clauses = []
    knowledgeBase = []
    functions = []
    predicates = []
    variables = []
    with open(fileName) as file:
        for line in file:
            if line.startswith("Predicates"):
                predicates.extend(line.split()[1:])
            elif line.startswith("Variables"):
                variables.extend(line.split()[1:])
            elif line.startswith("Constants"):
                constants.extend(line.split()[1:])
            elif line.startswith("Functions"):
                functions.extend(line.split()[1:])
            elif line.startswith("Clauses"):
                continue
            else:
                clauses.append(line.strip())
    for clause in clauses:
        currentClause = Clause()
        currentPredicates = []
        for term in clause.split():
            outerArgs = []
            if "(" in term and ")" in term:
                outerVals = term[term.find("(") + 1 : term.rfind(")")].split(",")
                for outerArg in outerVals:
                    if "(" in outerArg and ")" in outerArg:
                        innerArgs = []
                        innerVal = outerArg.split("(")[0]
                        innerVals = outerArg[outerArg.find('(') + 1 : outerArg.rfind(')')].split(",")
                        for innerArg in innerVals:
                            innerArgType = getValType(innerArg, constants, functions, variables)
                            currentArg = Term(innerArg, innerArgType)
                            innerArgs.append(currentArg)
                        valType = getValType(innerVal, constants, functions, variables)
                        currentArg = Term(innerVal, valType, innerArgs)
                        outerArgs.append(currentArg)
                    else:
                        outerArgType = getValType(outerArg, constants, functions, variables)
                        outerArgs.append(Term(outerArg, outerArgType))
            predicate = term.split("(")[0]
            if predicate in predicates:
                currentPredicates.append(Predicate(predicate, False, outerArgs))
            if predicate[1:] in predicates:
                if predicate[0] == '!':
                    currentPredicates.append(Predicate(predicate[1:], True, outerArgs))
        currentClause.setPredicates(currentPredicates)
    return knowledgeBase, constants, predicates, variables


def getValType(val, constants, functions, variables):
    if val in variables:
        return "Variable"
    elif val in constants:
        return "Constant"
    elif val in functions:
        return "Function"


def checkSatisfiability(knowledgeBase, constants, predicates, variables):
    unifiedResolver = set()
    while True:
        clausePairs = makePairs(knowledgeBase, True)
        for pair in clausePairs:
            resolver = set()
            if len(predicates) > 0 and len(constants) == 0 and len(variables) == 0:
                resolver = resolvePredicates(pair)
            elif len(predicates) > 0 and len(constants) > 0 and len(variables) == 0:
                resolver = resolveConstants(pair)
            else:
                resolver = resolveAll(pair, knowledgeBase)
            if resolver is None:
                return "no"
            elif len(resolver) > 0:
                unifiedResolver = performUnion(unifiedResolver, resolver)
        if checkForSubset(unifiedResolver, knowledgeBase):
            return "yes"
        knowledgeBase = performUnion(knowledgeBase, unifiedResolver)

def makePairs(clauses, flipSwitch):
    pairs = []
    for comb in combinations(clauses, 2):
        pairs.append(comb)

    if flipSwitch:
        pairsTemp = deepcopy(pairs)
        pairs = checkPair(pairsTemp, pairs)
        pairsTemp = deepcopy(pairs)
        for pair in pairsTemp:
            if str(pair[0]) in pair[1].seen or str(pair[1]) in pair[0].seen:
                pairs.remove(pair)
    else:
        pairs = checkPair(pairs, pairs)
    return pairs

def checkPair(pairsTemp, pairs):
    for pair in pairsTemp:
        firstClauseResolver = []
        for clause in pair[0].resolver:
            firstClauseResolver.append(str(clause))
        secondClauseResolver = []
        for clause in pair[1].resolver:
            secondClauseResolver.append(str(clause))
        if str(pair[0]) in secondClauseResolver or str(pair[1]) in firstClauseResolver:
            pairs.remove(pair)
    return pairs


def resolvePredicates(pair):
    resolvedIntoSet = set()
    firstClause = pair[0]
    secondClause = pair[1]
    predsInFirstTemp = firstClause.predicates.copy()
    predsInSecondTemp = secondClause.predicates.copy()

    for predsInFirst in firstClause.predicates:
        for predsInSecond in secondClause.predicates:
            if predsInFirst.name == predsInSecond.name and \
                    ((predsInFirst.negation and not predsInSecond.negation)
                     or (not predsInFirst.negation and predsInSecond.negation)):
                predsInFirstTemp.remove(predsInFirst)
                predsInSecondTemp.remove(predsInSecond)
                if len(predsInFirstTemp) == 0 and len(predsInSecondTemp) == 0:
                    return None

                resolvedInto = Clause(predsInFirstTemp + predsInSecondTemp)
                resolvedInto.resolver = set(pair)
                predsInFirstTemp.append(predsInFirst)
                predsInSecondTemp.append(predsInSecond)
                resolvedIntoSet.add(resolvedInto)
    return resolvedIntoSet


def resolveConstants(pair):
    firstClause = pair[0]
    secondClause = pair[1]
    resolvedIntoSet = resolvePair(firstClause, secondClause)
    return resolvedIntoSet

def performUnion(firstSet, secondSet):
    clausesInSetOne = set()
    clausesInSetTwo =  set()
    setOneMap = {}
    setTwoMap = {}

    for clause in firstSet:
        clausesInSetOne.add(str(clause))
        setOneMap[str(clause)] = clause
    for clause in secondSet:
        clausesInSetTwo.add(str(clause))
        setTwoMap[str(clause)] = clause
    unionSet = clausesInSetOne.union(clausesInSetTwo)

    returnVal = set()
    for clause in unionSet:
        if clause in clausesInSetOne:
            returnVal.add(setOneMap[clause])
        elif clause in clausesInSetTwo:
            returnVal.add(setTwoMap[clause])
    return returnVal


def resolveAll(pair, clausePairs):
    firstClause = pair[0]
    secondClause = pair[1]
    firstClauseTemp = deepcopy(firstClause)
    secondClauseTemp = deepcopy(secondClause)
    unifiedClauses = unifyClauses(firstClause, secondClause)
    if universalSubstitution:
        unifiedPairs = makePairs(unifiedClauses, False)
    else:
        unifiedPairs = makePairs(unifiedClauses, True)

    resolvedIntoSet = set()
    if len(unifiedPairs) == 0:
        resolvedIntoSet = resolvePair(firstClause, secondClause)
    else:
        for pair in unifiedPairs:
            resolvedIntoSet = resolvePair(pair[0], pair[1], )

        if str(firstClauseTemp) != str(firstClause):
            duplicate = False
            for clause in clausePairs:
                if str(clause) == str(firstClause):
                    duplicate = True
            if not duplicate:
                clausePairs.append(firstClause)

        if str(secondClauseTemp) != str(secondClause):
            duplicate = False
            for clause in clausePairs:
                if str(clause) == str(secondClause):
                    duplicate = True
            if not duplicate:
                clausePairs.append(secondClause)

    return resolvedIntoSet

def resolvePair(firstClause, secondClause):
    resolvedIntoSet = set()
    predsInFirstTemp = deepcopy(firstClause.predicates)
    predsInSecondTemp = deepcopy(secondClause.predicates)
    for predInFirst in firstClause.predicates:
        for predInSecond in secondClause.predicates:
            if predInFirst.name == predInSecond.name and sameArguments(predInFirst, predInSecond) and \
                    ((predInFirst.negation and not predInSecond.negation)
                     or (not predInFirst.negation and predInSecond.negation)):
                predsInFirstTemp.remove(predInFirst)
                predsInSecondTemp.remove(predInSecond)
                if len(predsInFirstTemp) == 0 and len(predsInSecondTemp) == 0:
                    return None

                resolvedInto = Clause(predsInFirstTemp + predsInSecondTemp)
                resolvedInto.resolver = performUnion(resolvedInto.resolver, set(firstClause))
                resolvedInto.resolver = performUnion(resolvedInto.resolver, firstClause.resolver)
                resolvedInto.resolver = performUnion(resolvedInto.resolver, set(secondClause))
                resolvedInto.resolver = performUnion(resolvedInto.resolver, secondClause.resolver)
                predsInFirstTemp.append(predInFirst)
                predsInSecondTemp.append(predInSecond)
                resolvedIntoSet.add(resolvedInto)

    return resolvedIntoSet, predsInFirstTemp, predsInSecondTemp

def checkForSubset(first, second):
    setOne = set(first)
    setTwo = set(second)
    clausesInSetOne = set()
    clausesInSetTwo = set()
    setOneMap = {}
    setTwoMap = {}

    for clause in setOne:
        clausesInSetOne.add(str(clause))
        setOneMap[str(clause)] = clause
    for clause in setTwo:
        clausesInSetTwo.add(str(clause))
        setTwoMap[str(clause)] = clause

    return clausesInSetOne.issubset(clausesInSetTwo)


def unifyClauses(firstClause, secondClause):
    predicatePairs = []
    for pair in itertools.permutations(firstClause.predicates, len(secondClause.predicates)):
        predicatePairs.extend(list(zip(pair, secondClause.predicates)))
    substitutions = {}
    for pair in predicatePairs:
        sub = findSubstitution(pair[0], pair[1])
        if len(sub) > 0:
            for key in sub.keys():
                if key in substitutions.keys():
                    substitutions[key].append(sub[key])
                elif key not in substitutions.keys():
                    substitutions[key] = [sub[key]]
                else:
                    substitutions.update(sub)

    substitutionsTemp = {}
    variableSubstitution = True
    for orig, subs in substitutions.items():
        if orig == "Not Found":
            continue
        elif orig.type == "Variable":
            for val in subs:
                if val.type != "Variable":
                    variableSubstitution = False
                    break
            if variableSubstitution:
                tempName =  "$" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))
                variableTemp = Term(tempName, "Variable")
                for s in subs:
                    substitutionsTemp[s] = [variableTemp]
                substitutionsTemp[orig] = [variableTemp]
    substitutions.update(substitutionsTemp)

    if len(firstClause.seen) == 0:
        firstClause.seen = [str(secondClause)]
    else:
        firstClause.seen.append(str(secondClause))
    if len(secondClause.seen) == 0:
        secondClause.seen = [str(firstClause)]
    else:
        secondClause.seen.append(str(firstClause))
    if variableSubstitution:
        unified = substitute(firstClause, secondClause, substitutions)
    else:
        unified = substituteUnifiedClause(firstClause, secondClause, substitutions)
        global universalSubstitution
        universalSubstitution = not variableSubstitution
    return unified

def findSubstitution(firstPred, secondPred):
    if type(firstPred) == Term or type(secondPred) == Term:
        if firstPred.type == "Constant" or firstPred.type == "Variable" or \
                secondPred.type == "Constant" or secondPred.type == "Variable":
            if firstPred.name == secondPred.name and firstPred.type == secondPred.type:
                return None
            elif firstPred.type == "Variable":
                return {firstPred : secondPred}
            elif secondPred.type == "Variable":
                return {secondPred : firstPred}
            elif firstPred.type == "Variable" and type(secondPred) != Term:
                for arg in secondPred.arguments:
                    if firstPred.name == arg.name:
                        return {"Not Found" : None}
                return {firstPred : secondPred}
            elif secondPred.type == "Variable" and type(firstPred) != Term:
                for arg in firstPred.arguments:
                    if secondPred.name == arg.name:
                        return {"Not Found" : None}
                return {secondPred : firstPred}
            else:
                return {"Not Found" : None}
    elif firstPred.name != secondPred.name:
        return {"Not Found": []}
    elif len(firstPred.arguments) != len(secondPred.arguments):
        return {"Not Found": []}

    substitutionMap = {}
    for i in range(len(firstPred.arguments)):
        substitution = findSubstitution(firstPred.arguments[i], secondPred.arguments[i])
        if substitution is not None:
            substitutionMap.update(substitution)
    return substitutionMap

def substitute(firstClause, secondClause, substitutions):
    unifiedClauses = []
    seenBefore = []
    seenBefore.append(firstClause)
    seenBefore.append(secondClause)
    for clause in (firstClause, secondClause):
        clauseTemp = deepcopy(clause)
        for orig, subs in substitutions.items():
            if orig == "Not Found":
                continue
            else:
                for sVal in subs:
                    argumentsUpdate(sVal, clauseTemp, orig)
        unifiedClauses = checkDuplicate(clauseTemp, clause, unifiedClauses, seenBefore)
    return unifiedClauses

def substituteUnifiedClause(firstClause, secondClause, substitutions):
    unifiedClauses = []
    for clause in (firstClause, secondClause):
        for orig, subs in substitutions.items():
            if orig == "Not Found":
                continue
            else:
                for sVal in subs:
                    clauseTemp = deepcopy(clause)
                    argumentsUpdate(sVal, clauseTemp, orig)
                    unifiedClauses = checkDuplicate(clauseTemp, clause, unifiedClauses)
    return unifiedClauses

def argumentsUpdate(s, clauseTemp, orig):
    for pred in clauseTemp.predicates:
        for i in range(len(pred.arguments)):
            arg = pred.arguments[i]
            if arg.name == orig.name:
                pred.arguments = pred.arguments[:i]
                pred.arguments.append(s)
                pred.arguments.extend(pred.arguments[i + 1:])
            elif arg.type == "Function":
                for j in range(len(arg.arguments)):
                    if arg.arguments[j].name == orig.name:
                        arg.arguments = arg.arguments[:j]
                        arg.arguments.append(s)
                        arg.arguments.extend(arg.arguments[j + 1:])

def checkDuplicate(clauseTemp, clause, unifiedClauses, seenBefore = None):
    if str(clauseTemp) != str(clause):
        if seenBefore:
            clauseTemp.seen = seenBefore
        unifiedClauses.append(clauseTemp)
    else:
        duplicate = False
        for uc in unifiedClauses:
            if str(uc) == str(clauseTemp):
                duplicate = True
        if not duplicate:
            unifiedClauses.append(clause)
    return unifiedClauses

def sameArguments(predsInFirst, predsInSecond):
    argsOne = []
    argsTwo = []
    for val in predsInFirst.arguments:
        argsOne.append(str(val))
    for val in predsInSecond.arguments:
        argsTwo.append(str(val))
    return argsOne == argsTwo


if __name__ == '__main__':
    global universalSubstitution
    universalSubstitution = False
    fileName = sys.argv[1]
    # fileName = 'truck.cnf.txt'
    knowledgeBase, constants, predicates, variables = getValuesFromFile(fileName)
    print(checkSatisfiability(knowledgeBase, constants, predicates, variables))