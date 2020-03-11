from pprint import pprint

x_test = [1, 1, 1]
#   x1, x2, x3
X = [
    [1, 1, 0],  # 1
    [1, 0, 1],  # 1
    [1, 0, 0],  # 0
    [0, 1, 0]   # 0
]

y = [1, 1, 0, 0]
header = ['x1', 'x2', 'x3']

def main():
    uniq_vals = unique_vals(y)
    pprint(uniq_vals)
    counts = class_counts(y)
    pprint(counts)
    q = Question(1)
    print(q)
    print(q.match(X[0]))
    print(q.match(X[1]))
    true_rows, true_rows_y, false_rows, false_rows_y = partitions(X, y, q)
    pprint(true_rows)
    pprint(false_rows)
    g1 = gini(true_rows, true_rows_y)
    g2 = gini(false_rows, false_rows_y)
    g3 = gini(X, y)
    print(g1, g2, g3)
    gain = info_gain(g3, true_rows, true_rows_y, false_rows, false_rows_y)
    print(gain)
    best_gain, best_question = find_best_split(X, y)
    print(best_question)
    print(best_gain)
    print('*' * 20)
    tree = build_tree(X, y)
    res = classify(tree, x_test)
    print(res)


def classify(tree, example):
    curr_node = tree
    while True:
        if isinstance(curr_node, Leaf):
            return curr_node.predictions
        if curr_node.question.match(example):
            curr_node = curr_node.true_node
        else:
            curr_node = curr_node.false_node


def build_tree(X, y):
    best_gain, best_question = find_best_split(X, y)
    true_rows, true_rows_y, false_rows, false_rows_y = partitions(X, y, best_question)
    root = DecisionNode(best_question)
    root.true_data = [true_rows, true_rows_y]
    root.false_data = [false_rows, false_rows_y]
    nodes = [root]
    while True:
        new_nodes = []
        for node in nodes:
            gain, question = find_best_split(*node.true_data)
            if gain == 0:
                node.true_node = Leaf(node.true_data[1])
            else:
                new_node = DecisionNode(question)
                true_rows, true_rows_y, false_rows, false_rows_y = partitions(*node.true_data, question)
                new_node.true_data = [true_rows, true_rows_y]
                new_node.false_data = [false_rows, false_rows_y]
                node.true_node = new_node
                new_nodes.append(new_node)
            gain, question = find_best_split(*node.false_data)
            if gain == 0:
                node.false_node = Leaf(node.false_data[1])
            else:
                new_node = DecisionNode(question)
                true_rows, true_rows_y, false_rows, false_rows_y = partitions(*node.false_data, question)
                new_node.true_data = [true_rows, true_rows_y]
                new_node.false_data = [false_rows, false_rows_y]
                node.false_node = new_node
                new_nodes.append(new_node)
        nodes = new_nodes
        if not nodes:
            break
    return root

class DecisionNode:

    def __init__(self, question, true_node=None, false_node=None):
        self.question = question
        self.true_node = true_node
        self.false_node = false_node
        self.true_data = None
        self.false_data = None

class Leaf:

    def __init__(self, y):
        self.predictions = class_counts(y)

def find_best_split(X, y):
    features_num = len(X[0])
    curr_unc = gini(X, y)
    best_gain, best_question = 0, None
    for feature in range(features_num):
        values = set([xi[feature] for xi in X])
        for val in values:
            question = Question(feature)  # hmmm.....
            true_rows, true_rows_y, false_rows, false_rows_y = partitions(X, y, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(curr_unc, true_rows, true_rows_y, false_rows, false_rows_y)
            if gain >= best_gain:
                best_gain = gain
                best_question = question
    return best_gain, best_question


def info_gain(current_unc, left_x, left_y, right_x, right_y):
    p = len(left_x) / (len(left_x) + len(right_x))
    return current_unc - p * gini(left_x, left_y) - (1 - p) * gini(right_x, right_y)


def gini(X, y):
    counts = class_counts(y)
    impurity = 1
    examples_num = len(X)
    for lbl in counts:
        prob_of_lbl = counts[lbl] / examples_num
        impurity -= prob_of_lbl ** 2
    return impurity

def partitions(X, y, question):
    true_rows, false_rows = [], []
    true_rows_y, false_rows_y = [], []
    for xi, yi in zip(X, y):
        if question.match(xi):
            true_rows.append(xi)
            true_rows_y.append(yi)
        else:
            false_rows.append(xi)
            false_rows_y.append(yi)
    return true_rows, true_rows_y, false_rows, false_rows_y


def unique_vals(y):
    return set(y)

def class_counts(y):
    counts = {}
    for el in y:
        if el not in counts:
            counts[el] = 0
        counts[el] += 1
    return counts


class Question:

    def __init__(self, column, value=0):
        self.column = column
        self.value = value

    def match(self, row):
        return row[self.column] > 0

    def __repr__(self):
        return f"Is {header[self.column]} > 0 ?"


if __name__ == '__main__':
    main()