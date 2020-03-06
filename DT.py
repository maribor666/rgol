from pprint import pprint

header = ["color", "diameter", "label"]
train_data = [
    ['Green', 3, 'Apple'],  # 0
    ['Yellow', 3, 'Apple'], # 1
    ['Red', 1, 'Grape'],    # 2
    ['Red', 1, 'Grape'],    # 3
    ['Yellow', 3, 'Lemon']  # 4
]

def main():
    uniq_vals = unique_vals(train_data, 0)
    pprint(uniq_vals)
    counts = class_count(train_data)
    pprint(counts)
    q = Question(0, 'Green')
    print(q)
    example = train_data[0]
    res = q.match(example)
    print(res)
    true_rows, false_rows = partitions(train_data, Question(0, 'Green'))
    pprint(true_rows)
    pprint(false_rows)
    curr_unc = gini(train_data)
    print(curr_unc)
    ig = info_gain(true_rows, false_rows, curr_unc)
    print(ig)
    gain, question = find_best_split(train_data)
    print(question)
    print('*' * 20)
    my_tree = build_tree(train_data)
    print_tree(my_tree)
    res = classify(train_data[4], my_tree)
    print(res)


def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partitions(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


class Leaf:

    def __init__(self, rows):
        self.predictions = class_count(rows)

class Decision_Node:

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def find_best_split(rows):
    best_gain = 0
    best_question = None
    curr_unc = gini(rows)
    n_features = len(rows[0]) - 1
    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partitions(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, curr_unc)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


def gini(rows):
    counts = class_count(rows)
    examples_num = float(len(rows))
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / examples_num
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def partitions(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def class_count(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def is_nummeric(value):
    return isinstance(value, int) or isinstance(value, float)


def unique_vals(rows, col):
    return set([row[col] for row in rows])


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if is_nummeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        condition = '=='
        if is_nummeric(self.value):
            condition = '>='
        return f'Is {header[self.column]} {condition} {str(self.value)}?'


def print_tree(node, spacing=""):
    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return
    print(spacing + str(node.question))
    print(spacing + '--> True:')
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + '--> False:')
    print_tree(node.false_branch, spacing + "  ")



if __name__ == '__main__':
    main()